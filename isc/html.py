# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
from dataclasses import dataclass
from typing import List, Dict
from binascii import b2a_base64
from PIL import Image as PILImage
import io


@dataclass
class ImagePair:
    query: str
    db: str
    score: float
    correct: bool


def create_html_img_embed(
    im_id: str,
    ids_to_uri: Dict[str, str],
    max_w: int = 320,
    max_h: int = 320,
    color=None,
):
    style = ""
    if color is not None:
        style = f"""style="border: 5px solid {color}" """

    path = ids_to_uri[im_id]
    im = PILImage.open(path)
    im = im.convert(mode="RGB")
    im.thumbnail((max_w, max_h), PILImage.ANTIALIAS)
    buffer = io.BytesIO()
    im.save(buffer, format="JPEG")
    base64 = b2a_base64(buffer.getbuffer()).decode("ascii")
    html = f"""
    <figure {style}>
      <img src="data:image/jpeg;base64,{base64}" alt="{im_id}">
      <figcaption>{im_id}</figcaption>
    </figure>
    """
    return html


def create_html_pair(
    index: int,
    query_id: str,
    db_id: str,
    score: float,
    correct: bool,
    ids_to_uri: Dict[str, str],
    max_w: int = 320,
    max_h: int = 320,
):
    color = "#00ff99" if correct else "#ff3300"
    q_img = create_html_img_embed(
        query_id, ids_to_uri=ids_to_uri, max_w=max_w, max_h=max_h
    )
    db_img = create_html_img_embed(
        db_id, ids_to_uri=ids_to_uri, color=color, max_w=max_w, max_h=max_h
    )
    return f"""
     <tr>
        <td>{index}</td>
        <td>{q_img}</td>
        <td>{db_img}</td>
        <td>{score:.2f}</td>
      </tr>
    """


def create_pairs_html(
    pairs: List[ImagePair],
    ids_to_uri: Dict[str, str],
    max_w: int = 320,
    max_h: int = 320,
):
    rows = [
        create_html_pair(
            index=i,
            query_id=p.query,
            db_id=p.db,
            score=p.score,
            correct=p.correct,
            ids_to_uri=ids_to_uri,
            max_w=max_w,
            max_h=max_h,
        )
        for i, p in enumerate(pairs)
    ]
    rows = "\n".join(rows)
    return f"""
    <!DOCTYPE html>
    <html>
    <head></head>
    <body>
    <table>
      <tr>
        <th>id</th>
        <th>query</th>
        <th>db</th>
        <th>score</th>
      </tr>
      {rows}
    </table>
    </body>
    """
