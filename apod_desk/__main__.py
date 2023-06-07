"""Astronomy Picture of the Day Mac Desktop Setter"""

#!/usr/bin/env python3

import json
import logging
import os
import signal
import sys
import time
from datetime import date
from io import BytesIO
from random import randrange
from tempfile import NamedTemporaryFile
from dotenv import load_dotenv

from fake_useragent import UserAgent
import requests
import structlog
from AppKit import (
    NSApplication,
    NSBundle,
    NSObject,
    NSScreen,
    NSTimer,
    NSWorkspace,
)
from Foundation import NSURL
from PIL import Image, ImageDraw, ImageFont
from PyObjCTools import AppHelper

log = None
last_pic_files = {}
sleep_t = 600
ua = UserAgent()
NASA_API_KEY = ""


def build_logger(tty=True, name=__name__):
    logging.basicConfig(format="%(message)s", stream=sys.stdout)

    def logger_factory():
        logger = logging.getLogger(name)
        logger.setLevel(
            os.getenv("LOGLEVEL", logging.getLevelName(logging.INFO))
        )
        return logger

    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.dev.set_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if tty:
        processors.append(structlog.dev.ConsoleRenderer(force_colors=True))
    else:
        processors.append(structlog.processors.JSONRenderer())

    structlog.configure(
        processors=processors,
        logger_factory=logger_factory,
        context_class=dict,
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=False,
    )

    return structlog.get_logger()


def siginfo_handler(signum, frame):
    log.info("Forcing change of desktop image")
    set_desktop_image_periodically(None, None)


def construct_url(base, params):
    """Construct an APOD API URL"""
    url = base
    for i, (param, val) in enumerate(params):
        if i == 0:
            url += "?{0}={1}".format(param, val)
        else:
            url += "&{0}={1}".format(param, val)
    return url


def set_desktop_image_cocoa(img_file, screen, options=None):
    file_url = NSURL.fileURLWithPath_(img_file.name)

    # tell the workspace to set the desktop picture
    (
        result,
        error,
    ) = NSWorkspace.sharedWorkspace().setDesktopImageURL_forScreen_options_error_(
        file_url, screen, options, None
    )
    if error:
        log.error(error, img=img_file, screen=screen)


def set_desktop_image(img, img_desc):
    # make image options dictionary
    # we just make an empty one because the defaults are fine
    options = {}
    img_files = {}

    # iterate over all screens
    for screen in NSScreen.screens():
        screen_id = screen.deviceDescription()["NSScreenNumber"]
        sc_width, sc_height = (
            screen.frame().size.width,
            screen.frame().size.height,
        )
        (im_width, im_height) = img.size

        log.info(
            "Scaling image for display id",
            screen_id=screen_id,
            width=sc_width,
            height=sc_height,
        )
        ratio = max(sc_width / im_width, sc_height / im_height)
        try:
            img_2 = img.resize(
                (int(im_width * ratio), int(im_height * ratio)),
                resample=Image.Resampling.LANCZOS,
            )
        except OSError as e:
            log.error("Encountered exception", exc=str(e).strip())
            return

        img_2 = img_2.crop(box=(0, 0, sc_width, sc_height))
        new_w = img_2.size[0]
        new_h = img_2.size[1]
        log.info(
            "Image dimensions scaled and cropped",
            ratio=ratio,
            width=new_w,
            height=new_h,
        )

        draw = ImageDraw.Draw(img_2)
        draw.text(
            (new_w * 0.05, new_h * 0.95),
            img_desc,
            font=ImageFont.truetype("Times.ttc", 22),
            fill="rgb(255, 255, 255)",
            stroke_width=2,
            stroke_fill="black",
        )

        img_file = NamedTemporaryFile(suffix="*.png", delete=True)
        img_2.convert("RGB").save(img_file, format="PNG")  # for CYMK
        img_file.flush()

        set_desktop_image_cocoa(img_file, screen, options)

        img_files[screen_id] = {
            "file": img_file,
            "img": img_2,
            "desc": img_desc,
        }

    # You can't close, and delete the file before the next one
    # or else macOS will revert to the standard desktop image.
    for screen_id in list(last_pic_files):
        pic = last_pic_files[screen_id]
        if pic["file"] and not pic["file"].closed:
            pic["file"].close()
        last_pic_files.pop(screen_id)

    last_pic_files.update(img_files)


def retrieve_image(img_url, img_desc):
    headers = {
        'User-Agent': ua.random
    }

    try:
        r = requests.head(img_url, headers=headers, timeout=30)
    except requests.exceptions.ReadTimeout:
        return False

    if "content-type" in r.headers:
        c_type = r.headers["content-type"]
    else:
        log.warn("Missing content-type header, skipping")
        return False

    if "content-length" in r.headers:
        c_len = int(r.headers["content-length"])
        log.info("Content length", length=c_len)
    else:
        log.warn("Missing content-length header, skipping")
        return False

    inc = 1024
    start = 0
    end = inc
    image_data = b""
    retry = 15

    log.info("Checking image", hdurl=img_url, title=img_desc, type=c_type)

    valid_types = ["image/jpeg", "image/png"]
    if c_type not in valid_types:
        log.warn("Unhandled file-type", file_type=c_type)
        return False

    img = None

    Image.LOAD_TRUNCATED_IMAGES = True

    while len(image_data) <= c_len:
        headers = {
            "Range": f"bytes={start}-{end}",
            'User-Agent': ua.random
        }
        log.debug(headers)
        try:
            r_img = requests.get(img_url, headers=headers, timeout=30)
        except (
            requests.exceptions.ContentDecodingError,
            requests.exceptions.ChunkedEncodingError,
        ):
            log.warn(
                "Error decoding content from server. Skipping", url=img_url
            )
            return False
        except requests.exceptions.ReadTimeout:
            log.warn("Read timeout. Skipping.", url=img_url)
            return False

        image_data += r_img.content

        try:
            img = Image.open(BytesIO(image_data))
            log.info(
                "Image dimensions identified",
                width=img.size[0],
                height=img.size[1],
                start=start,
                end=end,
                len_img_data=len(image_data),
                c_len=c_len,
            )
        except (OSError, IOError) as e:
            if "cannot identify image file" in str(e):
                if retry > 0:
                    log.debug(
                        "Not yet identifed image dimensions",
                        start=start,
                        end=end,
                        retry=retry,
                    )
                elif retry == 0:
                    log.error("Image dimensions failed to be detected")
                    return False
            else:
                log.debug("Unexpected exception during decoding", err=str(e))

            start = end + 1
            end += inc if inc <= (c_len - 1) else (c_len - 1)
            retry -= 1
        except (UserWarning, Image.DecompressionBombError,
                requests.exceptions.RequestException) as e:
            log.error("Image retrieval failure", err=str(e))
            return False
        else:
            start = end + 1
            end = c_len - 1
            log.info(
                "Retrieving remainder of image data", start=start, end=end
            )
            headers = {
                "Range": f"bytes={start}-{end}",
                'User-Agent': ua.random
            }
            r_img = requests.get(img_url, headers=headers, timeout=30)
            image_data += r_img.content
            img = Image.open(BytesIO(image_data))
            break

    (width, height) = img.size
    if len(image_data) == c_len:
        log.info(
            "Got whole image and image is big enough",
            width=width,
            height=height,
            len_image_data=len(image_data),
            c_len=c_len,
        )

    if height < 2000 or width < 2000:
        log.warn(
            "Image is too small, move on",
            width=width,
            height=height,
        )
        return False

    set_desktop_image(img, img_desc)
    return True


def set_desktop_image_periodically(obj, notification):
    this_year = date.today().year
    base_url = "https://api.nasa.gov/planetary/apod"

    while True:
        log.info("-- MARK --")

        url_params = [("api_key", NASA_API_KEY), ("hd", True)]

        year = randrange(1997, this_year + 1)
        month = randrange(1, 13)
        day = randrange(1, 31)

        date_param = ("date", f"{year}-{month:02d}-{day:02d}")

        url_params.append(date_param)
        apod_url = construct_url(base_url, url_params)
        try:
            headers = {
                'User-Agent': ua.random
            }
            response = requests.get(apod_url, headers=headers, timeout=30)
            if response.status_code == 200:
                apod = json.loads(response.text)
                result = retrieve_image(apod["hdurl"], apod["title"])
                if result:
                    log.info(
                        "Setting image",
                        date=apod.get("date"),
                        hdurl=apod.get("hdurl"),
                        title=apod.get("title"),
                    )
                    log.info(
                        "Pausing for before next image.", sleep_time=sleep_t
                    )
                    break
            elif response.status_code != 200:
                backoff_sleep = 900
                log.warn(f"An unexpected server response was received: {response.status_code}",
                    sleep=backoff_sleep,
                )
                time.sleep(backoff_sleep)
        except KeyError:
            log.warn("No high definition image available, skipping.")
            #  time.sleep(10)
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.ReadTimeout,
        ):
            log.warn(
                "Conection error, or timeout, sleeping before retry",
                sleep=sleep_t,
            )
            time.sleep(sleep_t)
        except Exception as e:
            log.warn(
                f"Unhandled exception occurred ({e}): sleeping before retry",
                sleep=sleep_t,
            )
            time.sleep(sleep_t)


def set_desktop_image_by_notification(obj, notification):
    log.info("Space change detected")
    for screen in NSScreen.screens():
        screen_id = screen.deviceDescription()["NSScreenNumber"]
        if screen_id in last_pic_files:
            set_desktop_image_cocoa(last_pic_files[screen_id]["file"], screen)
        else:
            set_desktop_image_periodically(None, None)


def main():
    global log, NASA_API_KEY

    load_dotenv()

    NASA_API_KEY = os.getenv("NASA_API_KEY", "")

    if not NASA_API_KEY:
        raise ValueError("NASA_API_KEY is not set to a valid API key")

    log = build_logger(name="apod_desk")

    signal.signal(signal.SIGINFO, siginfo_handler)

    info = NSBundle.mainBundle().infoDictionary()
    info["LSBackgroundOnly"] = "1"

    NSApplication.sharedApplication()
    _cls = type("r", (NSObject,), {})
    _cls.spaceDidChange_ = set_desktop_image_by_notification
    _cls.getAndSetNewDesktop_ = set_desktop_image_periodically

    obsrv = _cls.new()

    NSWorkspace.sharedWorkspace().notificationCenter().addObserver_selector_name_object_(
        obsrv,
        "spaceDidChange:",
        "NSWorkspaceActiveSpaceDidChangeNotification",
        None,
    )

    obsrv.getAndSetNewDesktop_(None)
    NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
        sleep_t, obsrv, obsrv.getAndSetNewDesktop_, None, True
    )

    # AppHelper.runEventLoop()
    AppHelper.runConsoleEventLoop(installInterrupt=True)


if __name__ == "__main__":
    main()
