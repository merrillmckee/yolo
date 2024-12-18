import gdown


def download_gdrive_data(url, output):
    # url = "https://drive.google.com/drive/folders/1HWFHKCprFzR7H7TYhrE-W7v4bz2Vc7Ia"
    gdown.download_folder(
        url=url,
        output=output,
        quiet=False,
        use_cookies=False,
        remaining_ok=True,
    )


if __name__ == "__main__":
    # Summary: only downloads 50 files per gdrive folder :(
    #
    # Found gdown in a SO post on downloading gdrive files; I couldn't get other suggestions to work
    # Even this solution didn't seem to work on some of the initial attempts.
    #
    # did not help
    # - in the venv/lib/.../gdown, modify "download_folder.py" MAX_NUMBER_FILES = 5000000  # from 50
    # - update: on inspection, only 50 files per folder were downloaded after the change above
    #
    # may have helped?
    # - in the venv/lib/.../gdown, modify "download.py" by adding small sleep to get_url_from_gdrive_confirmation()
    # - in ~/.cache/gdown delete a possibly malformed "cookies.txt" file
    # - run with use_cookies=False
    #

    url_ = "https://drive.google.com/drive/folders/1SiafFkKtjbhMCwX7FPYA-FVKQ_gJ3NO-"
    output_ = "../../data/nba1022"
    download_gdrive_data(url_, output_)
