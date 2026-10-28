"""最小化的 Drive mock:只實作 drive.py 用到的 files().list/create/update/get_media。"""
import collections
import itertools
import re
import types

FOLDER = "application/vnd.google-apps.folder"


class _Req:
    def __init__(self, fn):
        self._fn = fn

    def execute(self):
        return self._fn()


class FakeFiles:
    def __init__(self, o):
        self.o = o

    def list(self, q, fields=None, pageSize=100, orderBy=None, pageToken=None, spaces=None):
        return _Req(lambda: self.o.do_list(q, pageSize, orderBy, pageToken))

    def create(self, body, media_body=None, fields=None):
        return _Req(lambda: self.o.do_create(body, media_body))

    def update(self, fileId, media_body=None, fields=None):
        return _Req(lambda: self.o.do_update(fileId, media_body))

    def get_media(self, fileId):
        return types.SimpleNamespace(file_id=fileId)


class FakeService:
    def __init__(self):
        self.items = []
        self.fail = {}                       # op -> [exception, ...],依序拋出
        self.calls = collections.Counter()
        self.max_page = 1000
        self._ids = itertools.count(1)
        self._clock = itertools.count(1)

    def files(self):
        return FakeFiles(self)

    def _enter(self, op):
        self.calls[op] += 1
        q = self.fail.get(op)
        if q:
            raise q.pop(0)

    def add(self, name, parent, folder=False, content=b""):
        it = {"id": f"id{next(self._ids)}", "name": name, "parents": [parent],
              "mimeType": FOLDER if folder else "text/plain",
              "createdTime": next(self._clock), "modifiedTime": next(self._clock),
              "content": content}
        self.items.append(it)
        return it

    def find(self, name, parent=None):
        return [i for i in self.items if i["name"] == name
                and (parent is None or parent in i["parents"])]

    @staticmethod
    def _media_bytes(media):
        return media.getbytes(0, media.size()) if media is not None else b""

    def do_list(self, q, page_size, order_by, page_token):
        self._enter("list")
        m = re.search(r"\bname='((?:[^'\$|\\.)*)'", q)
        name = re.sub(r"\$.)", r"\1", m.group(1)) if m else None
        parent = re.search(r"'([^']+)' in parents", q).group(1)
        want_folder = "mimeType!=" not in q
        res = [i for i in self.items
               if parent in i["parents"] and (name is None or i["name"] == name)
               and (i["mimeType"] == FOLDER) == want_folder]
        if order_by == "createdTime":
            res.sort(key=lambda i: i["createdTime"])
        elif order_by == "modifiedTime desc":
            res.sort(key=lambda i: i["modifiedTime"], reverse=True)
        start = int(page_token or 0)
        end = start + min(page_size, self.max_page)
        out = {"files": [{k: i[k] for k in ("id", "name", "createdTime", "modifiedTime")}
                         for i in res[start:end]]}
        if end < len(res):
            out["nextPageToken"] = str(end)
        return out

    def do_create(self, body, media):
        self._enter("create")
        it = self.add(body["name"], body["parents"][0],
                      folder=body.get("mimeType") == FOLDER,
                      content=self._media_bytes(media))
        return {"id": it["id"]}

    def do_update(self, file_id, media):
        self._enter("update")
        it = next(i for i in self.items if i["id"] == file_id)
        it["content"] = self._media_bytes(media)
        it["modifiedTime"] = next(self._clock)
        return {"id": file_id}


def make_downloader(svc):
    """取代 MediaIoBaseDownload:先寫入內容,再依 svc.fail['download'] 拋錯(模擬下載到一半失敗)。"""
    class DL:
        def __init__(self, fh, req):
            self.fh, self.req = fh, req

        def next_chunk(self):
            item = next(i for i in svc.items if i["id"] == self.req.file_id)
            self.fh.write(item["content"])
            svc._enter("download")
            return None, True
    return DL
