#!/usr/bin/env python3
import json
import re
import os.path
import pickle
import argparse
from datetime import datetime, timezone, timedelta
from collections import Counter
from itertools import chain
from multiprocessing import Pool
from operator import itemgetter
from copy import copy
from bisect import bisect_left

from sudachipy import tokenizer, dictionary
import jaconv

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.image as mpimg
from matplotlib.ticker import MultipleLocator, PercentFormatter
from matplotlib.font_manager import FontProperties

from adjustText import adjust_text

from emoji import EMOJI_DATA

matplotlib.use("module://mplcairo.macosx")

TIMELINE = os.path.join(os.path.dirname(__file__), "timeline.pickle")
TIMEZONE = timezone(timedelta(hours=9), "JST")

matplotlib.rcParams["font.sans-serif"] = ["Hiragino Maru Gothic Pro", "Yu Gothic", "Meirio", "Takao", "IPAexGothic", "IPAPGothic", "VL PGothic", "Noto Sans CJK JP"]
emoji_prop = FontProperties(fname="/System/Library/Fonts/Apple Color Emoji.ttc")

UNICODE_EMOJI = EMOJI_DATA.keys()

# (ward to plot, line style, color)
RTA_EMOTES = (
    ("rtaClap", "-", "#ec7087"),
    ("rtaPray", "-", "#f7f97a"),
    (("rtaGl", "GL"), "-", "#5cc200"),
    (("rtaGg", "GG"), "-", "#ff381c"),
    ("rtaCheer", "-", "#ffbe00"),
    ("rtaHatena", "-", "#ffb5a1"),
    ("rtaR", "-", "white"),
    (("rtaCry", "BibleThump"), "-", "#5ec6ff"),

    ("rtaListen", "-.", "#5eb0ff"),
    ("rtaKabe", "-.", "#bf927a"),
    ("rtaFear", "-.", "#8aa0ec"),
    (("rtaRedbull", "rtaRedbull2", "レッドブル"), "-.", "#98b0df"),
    # ("rtaPokan", "-.", "#838187"),
    # ("rtaGogo", "-.", "#df4f69"),
    # ("rtaBanana", ":", "#f3f905"),
    # ("rtaBatsu", ":", "#5aafdd"),
    # ("rtaShogi", ":", "#c68d46"),
    # ("rtaThink", ":", "#f3f905"),
    # ("rtaIizo", ":", "#0f9619"),
    ("rtaDeath", "-.", "#ffbe00"),
    ("rtaDaba", "-.", "white"),

    ("rtaHello", ":", "#ff3291"),
    # ("rtaHmm", "-.", "#fcc7b9"),
    ("rtaPog", ":", "#f8c900"),
    ("rtaMaru", ":", "#c80730"),
    ("rtaFire", ":", "#E56124"),
    ("rtaIce", ":", "#CAEEFA"),
    # ("rtaThunder", ":", "#F5D219"),
    # ("rtaPoison", ":", "#9F65B2"),
    ("rtaGlitch", ":", "#9F65B2"),
    # ("rtaWind", ":", "#C4F897"),
    # ("rtaOko", "-.", "#d20025"),
    # ("rtaWut", ":", "#d97f8d"),
    ("rtaPolice", ":", "#7891b8"),
    # ("rtaChan", "-.", "green"),
    # ("rtaKappa", "-.", "#ffeae2"),

    # ("rtaSleep", "-.", "#ff8000"),
    # ("rtaCafe", "--", "#a44242"),
    # ("rtaDot", "--", "#ff3291"),

    # ("rtaShi", ":", "#8aa0ec"),
    # ("rtaGift", ":", "white"),
    # ("rtaAnkimo", ":", "#f92218 "),

    # ("rtaFrameperfect", "--", "#ff7401"),
    # ("rtaPixelperfect", "--", "#ffa300"),
    (("草", "ｗｗｗ", "LUL"), "--", "#1e9100"),
    ("DinoDance", "--", "#00b994"),
    ("無敵時間", "--", "red"),
    ("Cheer（ビッツ）", "--", "#bd62fe"),
    ("石油王", "--", "yellow"),
    ("かわいい", "--", "#ff3291"),
    ("目隠し", "--", "#cccccc")
)
VOCABULARY = set(w for w, _, _, in RTA_EMOTES if isinstance(w, str))
VOCABULARY |= set(chain(*(w for w, _, _, in RTA_EMOTES if isinstance(w, tuple))))

EXCLUDE_MESSAGE_TERMS = (
    " subscribed with Prime",
    " subscribed at Tier ",
    " gifted a Tier ",
    " is gifting ",
    " raiders from "
)

# (title, movie start time as timestamp, offset hour, min, sec)
GAMES = (
    ("はじまりのあいさつ", 1691602246.542, 0, 32, 23, "right"),
    ("Celeste", 1691602246.542, 0, 35, 15),
    ("OMORI", 1691602246.542, 1, 46, 34),
    ("チュウリップ", 1691602246.542, 2, 50, 23),
    ("プリンス・オブ・ペルシャ", 1691602246.542, 4, 35, 14),
    ("Nuclear Blaze", 1691602246.542, 5, 34, 35),
    ("Stray", 1691602246.542, 6, 18, 47),
    ("モンスターハンターライズ", 1691602246.542, 7, 53, 6),
    ("バトルネットワーク ロックマンエグゼ２", 1691602246.542, 9, 6, 10),
    ("ロックマンX4", 1691602246.542, 11, 19, 28),
    ("片道勇者プラス", 1691602246.542, 12, 6, 22),
    ("Pokemon LEGENDS アルセウス", 1691602246.542, 13, 3, 46),
    ("Unrailed!", 1691602246.542, 17, 32, 52),
    ("星のカービィ 夢の泉の物語", 1691602246.542, 18, 39, 9),
    ("星のカービィ 鏡の大迷宮", 1691602246.542, 20, 4, 4),
    ("星のカービィWii デラックス", 1691602246.542, 21, 44, 39),
    ("プロゴルファー猿", 1691602246.542, 22, 42, 31),
    ("マリオゴルフ64", 1691602246.542, 23, 21, 34),
    ("ギボン: ジャングルを超えて", 1691602246.542, 24, 37, 57, "right"),
    ("Curse Crackers:\nFor Whom the Belle Toils", 1691602246.542, 25, 25, 3, "right"),
    ("グノーシア", 1691602246.542, 25, 44, 3, "right"),
    ("Lobotomy Corporation", 1691602246.542, 27, 48, 30),
    ("スプラトゥーン3", 1691602246.542, 30, 0, 30),
    ("マジカルドロップ\nトリロジー リレー", 1691602246.542, 31, 16, 38, "right"),
    ("ぷよぷよ4部作リレー", 1691602246.542, 31, 54, 37, "right"),
    ("The Pedestrian", 1691602246.542, 32, 41, 26),
    ("CALLING ～黒き着信～", 1691602246.542, 33, 20, 44),

    ("SPLATTERHOUSE", 1691727501.91, 0, 4, 52, "right"),
    ("愛・超兄貴", 1691727501.91, 0, 36, 50),
    ("チョコボの不思議なダンジョン2", 1691727501.91, 1, 3, 55),
    ("塊魂TRIBUTE", 1691727501.91, 3, 24, 50),
    ("ソードオブソダン", 1691727501.91, 5, 36, 43, "right"),
    ("ドラゴンズ\nレア", 1691727501.91, 5, 59, 55),
    ("Spark the Electric Jester 3", 1691727501.91, 6, 21, 49),
    ("ゼルダの伝説 ブレス オブ ザ ワイルド", 1691727501.91, 7, 50, 22),
    ("Ninja Gaiden Black", 1691727501.91, 9, 14, 59),
    ("BADLAND: GOTY Edition", 1691727501.91, 11, 12, 5),
    ("Vampire\nSurvivors", 1691727501.91, 12, 15, 51),
    ("極魔界村 改", 1691727501.91, 12, 43, 49),
    ("メタルギアソリッド3 スネークイーター", 1691727501.91, 13, 52, 42),
    ("アーマード・コア ネクサス", 1691727501.91, 16, 57, 8),
    ("常世ノ塔", 1691727501.91, 18, 10, 17),
    ("TUNIC", 1691727501.91, 18, 33, 43),
    ("スーパーマリオワールド", 1691727501.91, 19, 30, 30),
    ("東京バス案内2", 1691727501.91, 21, 10, 41),
    ("F-ZERO X EXPANSION KIT", 1691727501.91, 21, 36, 37),
    ("Jelly Drift", 1691727501.91, 22, 51, 42, "right"),
    ("電車でD\nLightning\nStage", 1691727501.91, 23, 2, 9),
    ("SIREN: New Translation", 1691727501.91, 23, 30, 18),

    ("Dead Space Remake(2023)", 1691813367.433, 1, 3, 13),
    ("Battlefield 4", 1691813367.433, 3, 53, 16),

    ("鬼武者3", 1691828102.937, 2, 19, 0),
    ("イース7", 1691828102.937, 3, 25, 10),
    ("ルーンファクトリー5", 1691828102.937, 5, 29, 3),
    ("ドラゴンクエスト2", 1691828102.937, 7, 31, 36, "right"),
    ("パネルクイズ アタック25", 1691828102.937, 10, 22, 11, "right"),
    ("Castlevania\n暁月の円舞曲", 1691828102.937, 10, 52, 9, "right"),
    ("悪魔城伝説", 1691828102.937, 11, 9, 49),
    ("BAYONETTA 3", 1691828102.937, 11, 48, 25),
    ("ドンキーコング64", 1691828102.937, 14, 18, 27),
    ("Simple Fish\nAdventure", 1691828102.937, 15, 36, 49),
    ("Getting\nOver It with\nBennett\nFoddy", 1691828102.937, 16, 12, 57),
    ("ゼルダの伝説 ムジュラの仮面3D", 1691828102.937, 16, 31, 29),
    ("ゼルダの伝説 時のオカリナ", 1691828102.937, 18, 47, 56),
    ("Ninjala", 1691828102.937, 20, 43, 40),
    ("Cult of the Lamb", 1691828102.937, 22, 7, 28),
    ("Devolver Bootleg", 1691828102.937, 23, 34, 57),
    ("GS美神 ～除霊師はナイスバディ～", 1691828102.937, 24, 7, 9),
    ("スーパーメトロイド", 1691828102.937, 25, 10, 51),
    ("アトミックロボキッド", 1691828102.937, 26, 23, 25, "right"),
    ("たまごっちの\nプチプチおみせっち", 1691828102.937, 26, 54, 9),
    ("魔界戦記ディスガイア2", 1691828102.937, 28, 28, 11),
    ("イースIII\nワンダラーズ・フロム・イース", 1691828102.937, 30, 16, 17),
    ("トーキョージャングル", 1691828102.937, 31, 11, 40),
    ("ベイグラントストーリー", 1691828102.937, 32, 27, 0),
    ("Pump It Up Infinity", 1691828102.937, 34, 36, 45),
    ("ボボボーボ・ボーボボ\n集まれ!!体感ボーボボ", 1691828102.937, 35, 47, 27),
    ("スーパーワギャンランド2", 1691828102.937, 36, 25, 55),
    ("ロックマン9 野望の復活!!", 1691828102.937, 37, 29, 46),
    ("Minecraft", 1691828102.937, 38, 25, 29),
    ("Trials Rising", 1691828102.937, 39, 1, 24),
    ("GeoGuessr", 1691828102.937, 39, 28, 12),
    ("ドラえもん", 1691828102.937, 39, 55, 9),
    ("東方風神録\n～ Mountain\nof Faith.", 1691828102.937, 40, 35, 28, "right"),
    ("バトルガレッガ\nRev.2016", 1691828102.937, 41, 7, 27),
    ("東方剛欲異聞\n～ 水没した沈愁地獄", 1691828102.937, 41, 36, 48),

    ("Ruina 廃都の物語", 1691979590.386, 0, 1, 48),
    ("ゼノブレイド ディフィニティブ・エディション", 1691979590.386, 1, 15, 36),
    ("Seiken Densetsu:\nFinal Fantasy Gaiden", 1691979590.386, 5, 29, 4),
    ("Alisa", 1691979590.386, 6, 26, 41),
    ("RESIDENT EVIL: REVELATIONS 2", 1691979590.386, 7, 21, 12),
    ("DARK SOULS REMASTERED", 1691979590.386, 9, 26, 0),
    ("ヨイヤミダンサーズ", 1691979590.386, 11, 14, 3, "right"),
    ("黄昏ニ眠ル街", 1691979590.386, 11, 48, 17),
    ("Rocksmith 2014", 1691979590.386, 12, 34, 20),
    ("SOUND VOLTEX EXCEED GEAR\n(コナステ版)", 1691979590.386, 13, 10, 43),
    ("Classical\nMusic\nMinesweeper", 1691979590.386, 14, 20, 56),
    ("サンリオワールド\nスマッシュボール！", 1691979590.386, 14, 42, 21),
    ("ドラゴンバスター", 1691979590.386, 15, 23, 44),
    ("ファイナルファンタジーIV", 1691979590.386, 15, 54, 12),
    ("ファイナルファンタジーVI", 1691979590.386, 18, 14, 4),
    ("ときめきメモリアル～forever with you～", 1691979590.386, 22, 1, 29),
    ("スーパーマリオ64", 1691979590.386, 23, 15, 54),
    ("閉幕のあいさつ", 1691979590.386, 24, 14, 38, "right")
)


class Game:
    def __init__(self, name, t, h, m, s, align="left"):
        self.name = name
        self.startat = datetime.fromtimestamp(t + h * 3600 + m * 60 + s).replace(tzinfo=TIMEZONE)
        self.align = align


GAMES = tuple(Game(*args) for args in GAMES)

WINDOWSIZE = 1
WINDOW = timedelta(seconds=WINDOWSIZE)
AVR_WINDOW = 60
PER_SECONDS = 60
FIND_WINDOW = 15
DOMINATION_RATE = 0.6
COUNT_THRESHOLD = 37.5

DPI = 200
ROW = 5
PAGES = 4
YMAX = 700
WIDTH = 3840
HEIGHT = 2160

FONT_COLOR = "white"
FRAME_COLOR = "white"
BACKGROUND_COLOR = "#3f6392"
FACE_COLOR = "#274064"
ARROW_COLOR = "#ffff79"
MESSAGE_FILL_COLOR = "#0b0d2e"
MESSAGE_EDGE_COLOR = "#0080ff"

BACKGROUND = "rijs.png"


class Message:
    _tokenizer = dictionary.Dictionary().create()
    _mode = tokenizer.Tokenizer.SplitMode.C

    pns = (
        "無敵時間",
        "石油王",
        "躊躇しないでください",
        "国境なき医師団",
        "ナイスセーブ",
        "ナイスセーヌ",
        "ハイプトレイン",
        "待機時間",
        "見損なったぞカーネル",
        "タイトル回収",
        "目眩を起こす",
        "おばあちゃん",
        "待ち時間",
        "お遍路",
        "まもも",
        "こっちみんな",
        "たおれるぞ",
        "ヤマメですか",
        "勝てればいいの",
        "しりげをにる",
        "将軍モグタン"
    )
    pn_patterns = (
        (re.compile("[\u30A1-\u30FF]+ケンカ"), "〜ケンカ"),
        (re.compile("[a-zA-Z]+[0-9]+"), "Cheer（ビッツ）"),
        (re.compile("世界[1１一]位?"), "世界一"),
        (re.compile("ヨシ！+"), "ヨシ！"),
        (re.compile("くぁwせdrftgyふじこlp"), "くぁｗせｄｒｆｔｇｙふじこｌｐ"),
        (re.compile("くぁｗせｄｒｆｔｇｙふじこｌｐ"), "くぁｗせｄｒｆｔｇｙふじこｌｐ"),
        (re.compile("めまいを起こす"), "目眩を起こす")
    )
    stop_words = (
        "Squid2",
        "する",
        ''
    )

    @classmethod
    def _tokenize(cls, text):
        return cls._tokenizer.tokenize(text, cls._mode)

    def __init__(self, raw):
        # self.name = raw["author"]["name"]

        if "emotes" in raw:
            self.emotes = set(e["name"] for e in raw["emotes"]
                              if e["name"] not in self.stop_words)
        else:
            self.emotes = set()
        self.datetime = datetime.fromtimestamp(int(raw["timestamp"]) // 1000000).replace(tzinfo=TIMEZONE)

        self.message = raw["message"]
        self.msg = set()

        message = self.message
        for emote in self.emotes:
            message = message.replace(emote, "")
        for stop in self.stop_words:
            message = message.replace(stop, "")
        message = re.sub(r"https?://[\w/:%#\$&\?\(\)~\.=\+\-]+", "", message)

        #
        for pattern, replace in self.pn_patterns:
            match = pattern.findall(message)
            if match:
                self.msg.add(replace)
                if pattern.pattern.startswith('^') and pattern.pattern.endswith('$'):
                    message = ''
                else:
                    for m in match:
                        message = message.replace(m, "")

        #
        for pn in self.pns:
            if pn in message:
                self.msg.add(pn)
                message = message.replace(pn, "")

        #
        message = jaconv.h2z(message)

        # (名詞 or 動詞) (+助動詞)を取り出す
        parts = []
        currentpart = None
        for m in self._tokenize(message):
            part = m.part_of_speech()[0]

            if currentpart:
                if part == "助動詞":
                    parts.append(m.surface())
                else:
                    self.msg.add(''.join(parts))
                    parts = []
                    if part in ("名詞", "動詞"):
                        currentpart = part
                        parts.append(m.surface())
                    else:
                        currentpart = None
            else:
                if part in ("名詞", "動詞"):
                    currentpart = part
                    parts.append(m.surface())

        if parts:
            self.msg.add(''.join(parts))

        #
        kusa = False
        for word in copy(self.msg):
            if set(word) & set(('w', 'ｗ')):
                kusa = True
                self.msg.remove(word)
        if kusa:
            self.msg.add("ｗｗｗ")

        message = message.strip()
        if not self.msg and message:
            self.msg.add(message)

    def __len__(self):
        return len(self.msg)

    @property
    def words(self):
        return self.msg | self.emotes


def _make_messages(raw_message):
    if "name" in raw_message["author"] and raw_message["author"]["name"] == "fossabot":
        return

    for term in EXCLUDE_MESSAGE_TERMS:
        if term in raw_message["message"]:
            return
    return Message(raw_message)


def _parse_chat(paths):
    messages = []
    for p in paths:
        with open(p) as f, Pool() as pool:
            j = json.load(f)
            messages += [msg for msg in pool.map(_make_messages, j, len(j) // pool._processes)
                         if msg is not None]

    timeline = []
    currentwindow = messages[0].datetime.replace(microsecond=0) + WINDOW
    _messages = []
    for m in messages:
        if m.datetime <= currentwindow:
            _messages.append(m)
        else:
            timeline.append((currentwindow, *_make_timepoint(_messages)))
            while True:
                currentwindow += WINDOW
                if m.datetime <= currentwindow:
                    _messages = [m]
                    break
                else:
                    timeline.append((currentwindow, 0, Counter()))

    if _messages:
        timeline.append((currentwindow, *_make_timepoint(_messages)))

    return timeline


def _make_timepoint(messages):
    total = len(messages)
    counts = Counter(_ for _ in chain(*(m.words for m in messages)))

    return total, counts


def _load_timeline(paths):
    if os.path.exists(TIMELINE):
        with open(TIMELINE, "rb") as f:
            timeline = pickle.load(f)
    else:
        timeline = _parse_chat(paths)
        with open(TIMELINE, "wb") as f:
            pickle.dump(timeline, f)

    return timeline


def _save_counts(timeline):
    _, _, counters = zip(*timeline)

    counter = Counter()
    for c in counters:
        counter.update(c)

    with open("words.tab", 'w') as f:
        for w, c in sorted(counter.items(), key=itemgetter(1), reverse=True):
            print(w, c, sep='\t', file=f)


def _plot(timeline, normarize):
    scales = False
    if normarize:
        x, totals, _ = tuple(zip(*timeline))

        breaks = [game.startat for game in GAMES]
        breaks = [bisect_left(x, b) for b in breaks]
        breaks = [0] + breaks + [len(x)]

        scales = np.array([])
        totals = moving_average(totals) * PER_SECONDS
        for begin, end in zip(breaks, breaks[1:]):
            max_msgs = max(totals[begin:end])
            scales = np.concatenate((scales, np.ones(end - begin) / max_msgs))

    for npage in range(1, 1 + PAGES):
        chunklen = int(len(timeline) / PAGES / ROW)

        fig = plt.figure(figsize=(WIDTH / DPI, HEIGHT / DPI), dpi=DPI)
        fig.patch.set_facecolor(BACKGROUND_COLOR)
        plt.rcParams["savefig.facecolor"] = BACKGROUND_COLOR
        ax = fig.add_axes((0, 0, 1, 1))
        background_image = mpimg.imread(BACKGROUND)
        ax.imshow(background_image)

        plt.subplots_adjust(left=0.07, bottom=0.05, top=0.92)

        for i in range(1, 1 + ROW):
            nrow = i + ROW * (npage - 1)
            f, t = chunklen * (nrow - 1), chunklen * nrow
            x, c, y = zip(*timeline[f:t])
            # _x = tuple(t.replace(tzinfo=None) for t in x)

            ax = fig.add_subplot(ROW, 1, i)
            scale = False if scales is False else scales[f:t]

            _plot_row(ax, x, y, c, i == 1, i == ROW, scale)

        fig.suptitle(f"RTA in Japan Summer 2023 チャット頻出スタンプ・単語 ({npage}/{PAGES})",
                     color=FONT_COLOR, size="x-large")

        desc = "" if scales is False else ", ゲームタイトルごとの最大値=100%"
        ytitle = f"単語 / 分 （同一メッセージ内の重複は除外{desc}）"
        fig.text(0.03, 0.5, ytitle,
                 ha="center", va="center", rotation="vertical", color=FONT_COLOR, size="large")
        fig.savefig(f"{npage}.png", dpi=DPI, transparent=True)
        plt.close()
        print(npage)


def moving_average(x, w=AVR_WINDOW):
    _x = np.convolve(x, np.ones(w), "same") / w
    return _x[:len(x)]


def _plot_row(ax, x, y, total_raw, add_upper_legend, add_lower_legend, scales):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M", tz=TIMEZONE))
    ax.xaxis.set_major_locator(mdates.HourLocator())
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(range(0, 60, 5)))

    if scales is not False:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    else:
        ax.yaxis.set_minor_locator(MultipleLocator(50))

    ax.set_facecolor(FACE_COLOR)

    for axis in ("top", "bottom", "left", "right"):
        ax.spines[axis].set_color(FRAME_COLOR)

    ax.tick_params(colors=FONT_COLOR, which="both")
    ax.set_xlim(x[0], x[-1])
    if scales is not False:
        ax.set_ylim(0, 1)
    else:
        ax.set_ylim(0, YMAX)
    # ax.set_ylim(25, 800)
    # ax.set_yscale('log')

    total = moving_average(total_raw) * PER_SECONDS
    if scales is not False:
        total *= scales
    total = ax.fill_between(x, 0, total, color=MESSAGE_FILL_COLOR,
                            edgecolor=MESSAGE_EDGE_COLOR, linewidth=0.3)

    for i, game in enumerate(GAMES):
        annoat = YMAX if scales is False else 1
        if x[0] <= game.startat <= x[-1]:
            ax.axvline(x=game.startat, color=ARROW_COLOR, linestyle=":")
            ax.annotate(game.name, xy=(game.startat, annoat), xytext=(game.startat, annoat * 0.85), verticalalignment="top",
                        color=FONT_COLOR, arrowprops=dict(facecolor=ARROW_COLOR, shrink=0.05), ha=game.align)

    # ys = []
    # labels = []
    # colors = []
    for words, style, color in RTA_EMOTES:
        if isinstance(words, str):
            words = (words, )
        _y = np.fromiter((sum(c[w] for w in words) for c in y), int)
        if not sum(_y):
            continue
        _y = moving_average(_y) * PER_SECONDS
        if scales is not False:
            _y *= scales
        # ys.append(_y)
        # labels.append("\n".join(words))
        # colors.append(color if color else None)
        ax.plot(x, _y, label="\n".join(words), linestyle=style, color=(color if color else None))
    # ax.stackplot(x, ys, labels=labels, colors=colors)

    #
    avr_10min = moving_average(total_raw, FIND_WINDOW) * FIND_WINDOW
    words = Counter()
    for counter in y:
        words.update(counter)
    words = set(k for k, v in words.items() if v >= COUNT_THRESHOLD)
    words -= VOCABULARY

    annotations = []
    for word in words:
        at = []
        _ys = moving_average(np.fromiter((c[word] for c in y), int), FIND_WINDOW) * FIND_WINDOW
        for i, (_y, total_y) in enumerate(zip(_ys, avr_10min)):
            if _y >= total_y * DOMINATION_RATE and _y >= COUNT_THRESHOLD:
                ypoint = _y * PER_SECONDS / FIND_WINDOW * DOMINATION_RATE
                if scales is not False:
                    ypoint *= scales[i]
                at.append((i, ypoint))
        if at:
            at.sort(key=lambda x: x[1])
            at = at[-1]

            if any(c in UNICODE_EMOJI for c in word):
                text = ax.text(x[at[0]], at[1], word, color=FONT_COLOR, fontsize="xx-small", fontproperties=emoji_prop)
            else:
                text = ax.text(x[at[0]], at[1], word, color=FONT_COLOR, fontsize="xx-small")
            annotations.append(text)
    if annotations:
        adjust_text(annotations, only_move={"text": 'x'})

    if add_upper_legend:
        leg = ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", framealpha=0)
        _set_legend(leg)

    if add_lower_legend:
        leg = plt.legend([total], ["メッセージ / 分"], loc=(1.015, 0.4), framealpha=0)
        _set_legend(leg)
        msg = "図中の単語は{}秒間で{}%の\nメッセージに含まれていた単語\n({:.1f}メッセージ / 秒 以上のもの)".format(
            FIND_WINDOW, int(DOMINATION_RATE * 100), COUNT_THRESHOLD / FIND_WINDOW
        )
        plt.gcf().text(0.915, 0.06, msg, fontsize="x-small", color=FONT_COLOR)


def _set_legend(leg):
    frame = leg.get_frame()
    # frame.set_facecolor(FACE_COLOR)
    frame.set_edgecolor(FRAME_COLOR)

    for text in leg.get_texts():
        text.set_color(FONT_COLOR)


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json", nargs="+")
    parser.add_argument("-n", "--normarize", action="store_true")
    args = parser.parse_args()

    timeline = _load_timeline(args.json)
    _save_counts(timeline)

    _plot(timeline, args.normarize)


if __name__ == "__main__":
    _main()
