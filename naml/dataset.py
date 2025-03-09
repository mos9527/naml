from naml.modules import tqdm
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor, Future
import os, requests, time, json, hashlib, logging

logger = logging.getLogger(__name__)


@dataclass
class DatasetRemote:
    name: str
    url: str
    md5sum: str | None = None

    def __hash__(self):
        return hash(self.name) ^ hash(self.url) ^ hash(self.md5sum)

    def __eq__(self, other):
        return hash(self) == hash(other)


@dataclass
class DatasetLocal:
    path: str
    last_updated: float

    def remove(self):
        if os.path.exists(self.path):
            if os.path.isfile(self.path):
                os.remove(self.path)
            else:
                os.rmdir(self.path)

    def __eq__(self, value):
        raise NotImplementedError("DatasetLocal is not comparable")

    def readlines(self, encoding="utf-8"):
        with open(self.path, "r", encoding=encoding) as f:
            return f.readlines()


class Datasets(requests.Session):
    DB_NAME = ".naml.json"
    database: dict[str, DatasetLocal]

    __executor: ThreadPoolExecutor
    __futures: list[Future]
    __stoarge: str = None

    def __init__(self, storage_path: str = None, max_workers=4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.database = dict()
        self.__executor = ThreadPoolExecutor(max_workers=max_workers)
        self.__futures = []
        self.__stoarge = os.environ.get("NAML_DATASET_STORAGE", None) or storage_path
        self.load()

    @property
    def storage(self):
        path = self.__stoarge
        assert (
            path
        ), "Neither env `NAML_DATASET_STORAGE`, or keyword argument `storage` is set"
        path = os.path.expanduser(path)
        os.makedirs(path, exist_ok=True)
        return path

    @property
    def temp(self):
        path = os.path.join(self.storage, ".temp")
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def __download(self: "Datasets", remote: DatasetRemote):
        path = os.path.join(self.temp, remote.name)
        try:
            resp = self.get(remote.url, stream=True)
            resp.raise_for_status()
            size = int(resp.headers.get("content-length", 0))
            assert size, "Bad Content-Length"
            md5 = hashlib.md5()
            with open(path, "wb") as f:
                with tqdm(
                    total=size, unit="B", unit_scale=True, desc=remote.name
                ) as pbar:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                        md5.update(chunk)
                        pbar.update(len(chunk))
            if remote.md5sum:
                print(remote.md5sum)
                assert md5.hexdigest() == remote.md5sum, "Bad MD5"
            opath = os.path.join(self.storage, remote.name)
            os.replace(path, opath)
            self.database[remote] = DatasetLocal(
                path=opath,
                last_updated=time.time(),
            )
        except Exception as e:
            if os.path.exists(path):
                os.remove(path)
            raise e

    def __delitem__(self, key: DatasetRemote):
        self.database[key].remove()
        del self.database[key]

    def __getitem__(self, key: DatasetRemote) -> DatasetLocal:
        return self.database[key]

    def __contains__(self, key: DatasetRemote):
        return key in self.database

    def __setitem__(self, *args):
        raise NotImplementedError("Use download() method instead")

    def clear(self):
        for key in self.database:
            self[key].remove()
        self.database.clear()

    def queue(self, remote: DatasetRemote, blocking=False) -> Future | None:
        """Download a dataset by remote, optionally blocking until completion.
        Use join() to wait for all downloads to complete.
        """
        if remote in self.database:
            return
        if blocking:
            self.__download(self, remote)
            self.save()
        else:
            future = self.__executor.submit(self.__download, self, remote)
            self.__futures.append(future)
            return future

    def fetch(self, key: DatasetRemote) -> DatasetLocal:
        """Fetch a dataset by key, downloading it if necessary.
        This is guaranteed to block until the dataset is available locally.
        """
        if not key in self:
            self.queue(key, blocking=True)
            self.save()
        return self[key]

    def join(self):
        for future in self.__futures:
            future.result()
        self.__futures.clear()
        self.save()

    def save(self):
        with open(os.path.join(self.storage, self.DB_NAME), "w", encoding="utf-8") as f:
            json.dump(
                [(asdict(k), asdict(v)) for k, v in self.database.items()],
                f,
                ensure_ascii=False,
                indent=4,
            )
        logger.info(f"Saved {len(self.database)} datasets to {self.storage}")

    def load(self):
        path = os.path.join(self.storage, self.DB_NAME)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    db = json.load(f)
                    for key, value in db:
                        # XXX: This can be recursively defined. Enough for our use case here though.
                        remote = DatasetRemote(**key)
                        self.database[remote] = DatasetLocal(**value)
                    logging.info(f"entries={path}")
            except Exception as e:
                logging.error(f"Failed to load dataset database: {e}")
