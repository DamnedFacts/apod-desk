#!/usr/bin/env python3

"""Astronomy Picture of the Day Mac Desktop Setter."""


import argparse
import json
import logging
import os
import signal
import sys
import time
from datetime import date
from io import BytesIO
from enum import Enum
from random import randrange
from tempfile import NamedTemporaryFile
from dotenv import load_dotenv
from typing import Dict, Any, List
from fake_useragent import UserAgent
import requests
import structlog
from structlog import wrap_logger
from AppKit import (
    NSApplication,
    NSBundle,
    NSObject,
    NSScreen,
    NSTimer,
    NSWorkspace,
)
from Foundation import NSURL
from PIL import Image, ImageFont, ImageDraw, ImageStat, ImageOps, ImageFilter
from PyObjCTools import AppHelper

# Global to store image file paths
last_pic_files: Dict[int, Dict[str, Any]] = {}

args = None
log = None
sleep_t = 600
ua = UserAgent()
NASA_API_KEY = ""

class APODResult(Enum):
    SUCCESS = 0
    HTTP_NON_SUCCESS = 11
    HTTP_CONNECTION_ERROR = 12
    UNEXPECTED_ERROR = 100

def retry_after_delay(delay, retry_function, *args, **kwargs):
    def retry_wrapper():
        retry_function(*args, **kwargs)

    # Offload to a background thread with delay
    objc.dispatch_after(
        objc.dispatch_time(0, delay * 1_000_000_000),
        objc.dispatch_get_global_queue(0, 0),
        retry_wrapper
    )

def build_logger(tty=True, name=__name__):
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=logging.INFO)

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


def set_desktop_image_cocoa(img_file: str, screen: Any, options: Dict[str, Any] | None = None) -> None:
    if not options:
        options = {}

    file_url = NSURL.fileURLWithPath_(img_file)

    # tell the workspace to set the desktop picture
    (
        result,
        error,
    ) = NSWorkspace.sharedWorkspace().setDesktopImageURL_forScreen_options_error_(
        file_url, screen, options, None
    )
    if error:
        log.error(error, img=img_file, screen=screen)


def set_desktop_image(img: Image, img_desc: str) -> None:
    options = {}
    img_files: Dict[int, Dict[str, Any]] = {}

    times_font = ImageFont.truetype("Times.ttc", 22)

    for screen in NSScreen.screens():
        screen_id = screen.deviceDescription()["NSScreenNumber"]
        sc_width, sc_height = screen.frame().size.width, screen.frame().size.height
        (im_width, im_height) = img.size

        log.info("Scaling image for display id", screen_id=screen_id,
                 width=sc_width, height=sc_height)

        fills_width = fills_height = False

        if ('resize_to_fit' in args and args.resize_to_fit) or 'mirror_blur' in args or 'color_avg' in args:
            if im_width / im_height > sc_width / sc_height:
                # Image aspect ratio is wider than the screen's aspect ratio
                ratio = sc_width / im_width
                fills_width = True
            else:
                # Image aspect ratio is taller than the screen's aspect ratio
                ratio = sc_height / im_height
                fills_height = True
        else:
            ratio = max(sc_width / im_width, sc_height / im_height)

        try:
            img_2 = img.resize(
                (int(im_width * ratio), int(im_height * ratio)), resample=Image.Resampling.LANCZOS)
        except OSError as e:
            log.error("Encountered exception", exc=str(e).strip())
            return

        new_w = img_2.size[0]
        new_h = img_2.size[1]
        log.info("Image dimensions scaled",
                 ratio=ratio, width=new_w, height=new_h)

        final_img = Image.new("RGB", (int(sc_width), int(sc_height)))

        edge_boxes_and_positions = []
        if 'color_avg' in args:
            edge_width = int(min(new_w, new_h) * args.color_avg)
            if not fills_height:
                edge_boxes_and_positions.extend([
                    ((0, 0, new_w, edge_width), "top"),
                    ((0, new_h - edge_width, new_w, new_h), "bottom")
                ])
            if not fills_width:
                edge_boxes_and_positions.extend([
                    ((0, 0, edge_width, new_h), "left"),
                    ((new_w - edge_width, 0, new_w, new_h), "right")
                ])
        elif 'mirror_blur' in args and args.mirror_blur:
            edge_width = (int(sc_height) -
                          new_h) // 2 if not fills_height else 0
            edge_height = (int(sc_width) -
                           new_w) // 2 if not fills_width else 0
            if edge_width > 0:
                edge_boxes_and_positions.extend([
                    ((0, 0, new_w, edge_width), "top"),
                    ((0, new_h - edge_width, new_w, new_h), "bottom")
                ])
            if edge_height > 0:
                edge_boxes_and_positions.extend([
                    ((0, 0, edge_height, new_h), "left"),
                    ((new_w - edge_height, 0, new_w, new_h), "right")
                ])

        draw = ImageDraw.Draw(final_img)

        for box, position in edge_boxes_and_positions:
            if 'color_avg' in args:
                edge_img = img_2.crop(box)
                avg_color = ImageStat.Stat(edge_img).mean
                avg_color = tuple(map(int, avg_color))
            elif 'mirror_blur' in args and args.mirror_blur:
                edge_img = img_2.crop(box)
                if position in ["top", "bottom"]:
                    edge_img = ImageOps.flip(edge_img)
                else:
                    edge_img = ImageOps.mirror(edge_img)
                edge_img = edge_img.filter(ImageFilter.GaussianBlur(radius=50))

            if position == "top":
                draw.rectangle([(0, 0), (int(sc_width), (int(
                    sc_height) - new_h) // 2)], fill=avg_color if 'color_avg' in args else None)
                if 'mirror_blur' in args and args.mirror_blur:
                    final_img.paste(edge_img, (0, 0))
            elif position == "bottom":
                draw.rectangle([(0, int(sc_height) + (int(sc_height) - new_h) // 2), (int(
                    sc_width), int(sc_height))], fill=avg_color if 'color_avg' in args else None)
                if 'mirror_blur' in args and args.mirror_blur:
                    final_img.paste(edge_img, (0, int(sc_height) - edge_width))
            elif position == "left":
                draw.rectangle([(0, 0), ((int(sc_width) - new_w) // 2, int(sc_height))],
                               fill=avg_color if 'color_avg' in args else None)
                if 'mirror_blur' in args and args.mirror_blur:
                    final_img.paste(edge_img, (0, 0))
            elif position == "right":
                draw.rectangle([((int(sc_width) + new_w) // 2, 0), (int(sc_width),
                               int(sc_height))], fill=avg_color if 'color_avg' in args else None)
                if 'mirror_blur' in args and args.mirror_blur:
                    final_img.paste(
                        edge_img, ((int(sc_width) + new_w) // 2, 0))

        final_img.paste(img_2, ((int(sc_width) - new_w) //
                        2, (int(sc_height) - new_h) // 2))

        draw = ImageDraw.Draw(final_img)
        draw.text((new_w * 0.05, new_h * 0.95), img_desc, font=times_font,
                  fill="rgb(255, 255, 255)", stroke_width=2, stroke_fill="black")

        img_file = NamedTemporaryFile(suffix="*.png", delete=True)
        final_img.convert("RGB").save(img_file, format="PNG")  # for CYMK
        img_file.flush()

        set_desktop_image_cocoa(img_file.name, screen, options)

        img_files[screen_id] = {
            "file": img_file,
            "img": final_img,
            "desc": img_desc,
        }

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
    """
    Sets the desktop image for all screens.

    Parameters:
    img (PIL.Image): The image to set as the desktop image.
    img_desc (str): The description to draw on the image.
    font_path (str): The path to the font to use for the description text. Default is "Times.ttc".
    font_size (int): The font size to use for the description text. Default is 22.
    """

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
                    return APODResult.SUCCESS
            elif response.status_code != 200:
                log.warn("An unexpected server response was received",
                         sleep=backoff_sleep, status_code=response.status_code,
                         url=apod_url
                         )
                return APODResult.HTTP_NON_SUCCESS
        except KeyError:
            log.warn("No high definition image available, skipping.", url=apod_url)
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.ReadTimeout,
        ):
            log.warn(
                "Conection error, or timeout, sleeping before retry",
                sleep=sleep_t, url=apod_url
            )
            return APODResult.HTTP_CONNECTION_ERROR
        except Exception as e:
            log.warn(
                f"Unhandled exception occurred, sleeping before retry",
                sleep=sleep_t, url=apod_url, exc=e
            )
            return APODResult.UNEXPECTED_ERROR


def set_desktop_image_by_notification(obj, notification):
    log.info("Space change detected")
    for screen in NSScreen.screens():
        screen_id = screen.deviceDescription()["NSScreenNumber"]
        if screen_id in last_pic_files:
            set_desktop_image_cocoa(
                last_pic_files[screen_id]["file"].name, screen)
        else:
            set_desktop_image_periodically(None, None)


def main():
    global args, log, NASA_API_KEY

    # Define command-line arguments
    parser = argparse.ArgumentParser(description='NASA APOD Desktop')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--color_avg',
                       type=float,
                       nargs='?',
                       const=0.1,
                       default=argparse.SUPPRESS,
                       help=('Enable color averaging with an optional value '
                             'that represents the percent of the image from '
                             'the image edge to use for color averaging. '
                             'Default value if no value is provided is 10%'))
    group.add_argument('--mirror_blur',
                       action='store_true',
                       help='Enable mirroring and blurring of the image for the bars')
    parser.add_argument('--resize_to_fit',
                        action='store_true',
                        help='Enable resize-to-fit of the image such that the '
                             'longest dimension is proportionally shrunk to '
                             'fit within the screen\'s resolution')

    # Parse the command-line arguments
    args = parser.parse_args()

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
