import contextlib
from pathlib import Path
import shutil
import subprocess
from urllib import parse, request
from zipfile import BadZipFile, ZipFile, is_zipfile

import requests
import torch
from tqdm import tqdm

from juxtapose.utils import LOGGER, checks, clean_url, emojis, is_online, url2file


def is_url(url, check=True):
    """Check if string is URL and check if URL exists."""
    with contextlib.suppress(Exception):
        url = str(url)
        result = parse.urlparse(url)
        assert all([result.scheme, result.netloc])  # check if is url
        if check:
            with request.urlopen(url) as response:
                return response.getcode() == 200  # check if exists online
        return True
    return False


def unzip_file(file, path=None, exclude=(".DS_Store", "__MACOSX"), exist_ok=False):
    """
    Unzips a *.zip file to the specified path, excluding files containing strings in the exclude list.

    If the zipfile does not contain a single top-level directory, the function will create a new
    directory with the same name as the zipfile (without the extension) to extract its contents.
    If a path is not provided, the function will use the parent directory of the zipfile as the default path.

    Args:
        file (str): The path to the zipfile to be extracted.
        path (str, optional): The path to extract the zipfile to. Defaults to None.
        exclude (tuple, optional): A tuple of filename strings to be excluded. Defaults to ('.DS_Store', '__MACOSX').
        exist_ok (bool, optional): Whether to overwrite existing contents if they exist. Defaults to False.

    Raises:
        BadZipFile: If the provided file does not exist or is not a valid zipfile.

    Returns:
        (Path): The path to the directory where the zipfile was extracted.
    """
    if not (Path(file).exists() and is_zipfile(file)):
        raise BadZipFile(f"File '{file}' does not exist or is a bad zip file.")
    if path is None:
        path = Path(file).parent  # default path

    # Unzip the file contents
    with ZipFile(file) as zipObj:
        file_list = [f for f in zipObj.namelist() if all(x not in f for x in exclude)]
        top_level_dirs = {Path(f).parts[0] for f in file_list}

        if len(top_level_dirs) > 1 or not file_list[0].endswith("/"):
            path = Path(path) / Path(file).stem  # define new unzip directory

        # Check if destination directory already exists and contains files
        extract_path = Path(path) / list(top_level_dirs)[0]
        if extract_path.exists() and any(extract_path.iterdir()) and not exist_ok:
            # If it exists and is not empty, return the path without unzipping
            LOGGER.info(f"Skipping {file} unzip (already unzipped)")
            return path

        for f in file_list:
            zipObj.extract(f, path=path)

    return path  # return unzip dir


def check_disk_space(
    url="https://ultralytics.com/assets/coco128.zip", sf=1.5, hard=True
):
    """
    Check if there is sufficient disk space to download and store a file.

    Args:
        url (str, optional): The URL to the file. Defaults to 'https://ultralytics.com/assets/coco128.zip'.
        sf (float, optional): Safety factor, the multiplier for the required free space. Defaults to 2.0.
        hard (bool, optional): Whether to throw an error or not on insufficient disk space. Defaults to True.

    Returns:
        (bool): True if there is sufficient disk space, False otherwise.
    """
    with contextlib.suppress(Exception):
        gib = 1 << 30  # bytes per GiB
        data = int(requests.head(url).headers["Content-Length"]) / gib  # file size (GB)
        total, used, free = (x / gib for x in shutil.disk_usage("/"))  # bytes
        if data * sf < free:
            return True  # sufficient space

        # Insufficient space
        text = (
            f"WARNING ⚠️ Insufficient free disk space {free:.1f} GB < {data * sf:.3f} GB required, "
            f"Please free {data * sf - free:.1f} GB additional disk space and try again."
        )
        if hard:
            raise MemoryError(text)
        else:
            LOGGER.warning(text)
            return False

            # Pass if error
    return True


def safe_download(
    url,
    file=None,
    dir=None,
    unzip=True,
    delete=False,
    curl=False,
    retry=3,
    min_bytes=1e0,
    progress=True,
):
    """
    Downloads files from a URL, with options for retrying, unzipping, and deleting the downloaded file.

    Args:
        url (str): The URL of the file to be downloaded.
        file (str, optional): The filename of the downloaded file.
            If not provided, the file will be saved with the same name as the URL.
        dir (str, optional): The directory to save the downloaded file.
            If not provided, the file will be saved in the current working directory.
        unzip (bool, optional): Whether to unzip the downloaded file. Default: True.
        delete (bool, optional): Whether to delete the downloaded file after unzipping. Default: False.
        curl (bool, optional): Whether to use curl command line tool for downloading. Default: False.
        retry (int, optional): The number of times to retry the download in case of failure. Default: 3.
        min_bytes (float, optional): The minimum number of bytes that the downloaded file should have, to be considered
            a successful download. Default: 1E0.
        progress (bool, optional): Whether to display a progress bar during the download. Default: True.
    """
    f = dir / url2file(url) if dir else Path(file)  # URL converted to filename
    if (
        "://" not in str(url) and Path(url).is_file()
    ):  # URL exists ('://' check required in Windows Python<3.10)
        f = Path(url)  # filename
    elif not f.is_file():  # URL and file do not exist
        assert dir or file, "dir or file required for download"
        f = dir / url2file(url) if dir else Path(file)
        desc = f"Downloading {clean_url(url)} to '{f}'"
        LOGGER.info(f"{desc}...")
        f.parent.mkdir(parents=True, exist_ok=True)  # make directory if missing
        check_disk_space(url)
        for i in range(retry + 1):
            try:
                if curl or i > 0:  # curl download with retry, continue
                    s = "sS" * (not progress)  # silent
                    r = subprocess.run(
                        [
                            "curl",
                            "-#",
                            f"-{s}L",
                            url,
                            "-o",
                            f,
                            "--retry",
                            "3",
                            "-C",
                            "-",
                        ]
                    ).returncode
                    assert r == 0, f"Curl return value {r}"
                else:  # urllib download
                    method = "torch"
                    if method == "torch":
                        torch.hub.download_url_to_file(url, f, progress=progress)
                    else:
                        from juxtapose.utils import TQDM_BAR_FORMAT

                        with request.urlopen(url) as response, tqdm(
                            total=int(response.getheader("Content-Length", 0)),
                            desc=desc,
                            disable=not progress,
                            unit="B",
                            unit_scale=True,
                            unit_divisor=1024,
                            bar_format=TQDM_BAR_FORMAT,
                        ) as pbar:
                            with open(f, "wb") as f_opened:
                                for data in response:
                                    f_opened.write(data)
                                    pbar.update(len(data))

                if f.exists():
                    if f.stat().st_size > min_bytes:
                        break  # success
                    f.unlink()  # remove partial downloads
            except Exception as e:
                if i == 0 and not is_online():
                    raise ConnectionError(
                        emojis(
                            f"❌  Download failure for {url}. Environment is not online."
                        )
                    ) from e
                elif i >= retry:
                    raise ConnectionError(
                        emojis(f"❌  Download failure for {url}. Retry limit reached.")
                    ) from e
                LOGGER.warning(
                    f"⚠️ Download failure, retrying {i + 1}/{retry} {url}..."
                )

    if unzip and f.exists() and f.suffix in ("", ".zip", ".tar", ".gz"):
        unzip_dir = dir or f.parent  # unzip to dir if provided else unzip in place
        LOGGER.info(f"Unzipping {f} to {unzip_dir.absolute()}...")
        if is_zipfile(f):
            unzip_dir = unzip_file(file=f, path=unzip_dir)  # unzip
        elif f.suffix == ".tar":
            subprocess.run(
                ["tar", "xf", f, "--directory", unzip_dir], check=True
            )  # unzip
        elif f.suffix == ".gz":
            subprocess.run(
                ["tar", "xfz", f, "--directory", unzip_dir], check=True
            )  # unzip
        if delete:
            f.unlink()  # remove zip
        return unzip_dir
