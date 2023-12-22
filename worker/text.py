import re

from typing import Any, Callable, Dict, List, Tuple

from common.conf import get_conf
conf = get_conf()

langid_languages = [
    "af",
    "am",
    "an",
    "ar",
    "as",
    "az",
    "be",
    "bg",
    "bn",
    "br",
    "bs",
    "ca",
    "cs",
    "cy",
    "da",
    "de",
    "dz",
    "el",
    "en",
    "eo",
    "es",
    "et",
    "eu",
    "fa",
    "fi",
    "fo",
    "fr",
    "ga",
    "gl",
    "gu",
    "he",
    "hi",
    "hr",
    "ht",
    "hu",
    "hy",
    "id",
    "is",
    "it",
    "ja",
    "jv",
    "ka",
    "kk",
    "km",
    "kn",
    "ko",
    "ku",
    "ky",
    "la",
    "lb",
    "lo",
    "lt",
    "lv",
    "mg",
    "mk",
    "ml",
    "mn",
    "mr",
    "ms",
    "mt",
    "nb",
    "ne",
    "nl",
    "nn",
    "no",
    "oc",
    "or",
    "pa",
    "pl",
    "ps",
    "pt",
    "qu",
    "ro",
    "ru",
    "rw",
    "se",
    "si",
    "sk",
    "sl",
    "sq",
    "sr",
    "sv",
    "sw",
    "ta",
    "te",
    "th",
    "tl",
    "tr",
    "ug",
    "uk",
    "ur",
    "vi",
    "vo",
    "wa",
    "xh",
    "zh",
    "zu",
]

class ChineseTextSplitter:
    def __init__(self, pdf: bool = False, sentence_size: int = 256, **kwargs):
        # super().__init__(**kwargs)
        self.pdf = pdf
        self.sentence_size = sentence_size

    def split_text1(self, text: str) -> List[str]:
        if self.pdf:
            text = re.sub(r"\n{3,}", "\n", text)
            text = re.sub('\s', ' ', text)
            text = text.replace("\n\n", "")
        sent_sep_pattern = re.compile('([﹒﹔﹖﹗．。！？]["’”」』]{0,2}|(?=["‘“「『]{1,2}|$))')  # del ：；
        sent_list = []
        for ele in sent_sep_pattern.split(text):
            if sent_sep_pattern.match(ele) and sent_list:
                sent_list[-1] += ele
            elif ele:
                sent_list.append(ele)
        return sent_list

    def split_text(self, text: str) -> List[str]:   ##此处需要进一步优化逻辑
        if self.pdf:
            text = re.sub(r"\n{3,}", r"\n", text)
            text = re.sub('\s', " ", text)
            text = re.sub("\n\n", "", text)

        text = re.sub(r'([;；.!?。！？\?])([^”’])', r"\1\n\2", text)  # 单字符断句符
        text = re.sub(r'(\.{6})([^"’”」』])', r"\1\n\2", text)  # 英文省略号
        text = re.sub(r'(\…{2})([^"’”」』])', r"\1\n\2", text)  # 中文省略号
        text = re.sub(r'([;；!?。！？\?]["’”」』]{0,2})([^;；!?，。！？\?])', r'\1\n\2', text)
        # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
        text = text.rstrip()  # 段尾如果有多余的\n就去掉它
        # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
        ls = [i for i in text.split("\n") if i]
        for ele in ls:
            if len(ele) > self.sentence_size:
                ele1 = re.sub(r'([,，.]["’”」』]{0,2})([^,，.])', r'\1\n\2', ele)
                ele1_ls = ele1.split("\n")
                for ele_ele1 in ele1_ls:
                    if len(ele_ele1) > self.sentence_size:
                        ele_ele2 = re.sub(r'([\n]{1,}| {2,}["’”」』]{0,2})([^\s])', r'\1\n\2', ele_ele1)
                        ele2_ls = ele_ele2.split("\n")
                        for ele_ele2 in ele2_ls:
                            if len(ele_ele2) > self.sentence_size:
                                ele_ele3 = re.sub('( ["’”」』]{0,2})([^ ])', r'\1\n\2', ele_ele2)
                                ele2_id = ele2_ls.index(ele_ele2)
                                ele2_ls = ele2_ls[:ele2_id] + [i for i in ele_ele3.split("\n") if i] + ele2_ls[
                                                                                                       ele2_id + 1:]
                        ele_id = ele1_ls.index(ele_ele1)
                        ele1_ls = ele1_ls[:ele_id] + [i for i in ele2_ls if i] + ele1_ls[ele_id + 1:]

                id = ls.index(ele)
                ls = ls[:id] + [i for i in ele1_ls if i] + ls[id + 1:]
        return ls
    
def split_text(s1:str, sentence_size:int=256) ->list:
    '''
        根据中文书写习惯，分割文本，并返回da
        doc: 输入待处理的文本
        sentence_size: 最大分段长度
    '''
    text_splitter = ChineseTextSplitter(sentence_size=sentence_size)
    ls = text_splitter.split_text(s1)
    
    ls.append(conf.negativate_class)      # 添加other 项目
    return ls


def mark_text(text:str, pattern:str=r'[A-Za-z]+') -> List[Dict]:
    '''
    识别输入文本，自动标记中英文。暂时只支持中英文。
    通过re正则识别，还可以通过模型推理，常用的有：1.langid；2.langdetect；3.fasttext
    中文：[\u4e00-\u9fa5]
    英文：[a-zA-Z]
    '''
    engs = re.findall(pattern, text)
    langs = []
    if engs:
        for item in engs:
            p1, p_left = text.split(item)
            if p1:
                langs.append((p1, 'ZH'))
            langs.append((item, 'EN'))
            text = p_left
    if text:
        langs.append((text, 'ZH'))
            
    return langs

def split_by_language(text: str, target_languages: list = ["zh", "ja", "en"]) -> list:
    '''
    安装语种分割文本，返回文本段、语种的list
    '''
    pattern = (
        r"[\!\"\#\$\%\&\'\(\)\*\+\,\-\.\/\:\;\<\>\=\?\@\[\]\{\}\\\\\^\_\`"
        r"\！？\。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」"
        r"『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘\'\‛\“\”\„\‟…‧﹏.]+"
    )
    sentences = re.split(pattern, text)

    pre_lang = ""
    start = 0
    end = 0
    sentences_list = []

    sorted_target_languages = sorted(target_languages)
    if sorted_target_languages in [["en", "zh"], ["en", "ja"], ["en", "ja", "zh"]]:
        new_sentences = []
        for sentence in sentences:
            new_sentences.extend(split_alpha_nonalpha(sentence))
        sentences = new_sentences

    for sentence in sentences:
        if check_is_none(sentence):
            continue

        lang = classify_language(sentence, target_languages)

        end += text[end:].index(sentence)
        if pre_lang != "" and pre_lang != lang:
            sentences_list.append((text[start:end], pre_lang))
            start = end
        end += len(sentence)
        pre_lang = lang
    sentences_list.append((text[start:], pre_lang))

    return sentences_list

def classify_language(text: str, target_languages: list = None) -> str:
    module = 'langid'
    if module == "fastlid" or module == "fasttext":
        from fastlid import fastlid, supported_langs

        classifier = fastlid
        if target_languages != None:
            target_languages = [
                lang for lang in target_languages if lang in supported_langs
            ]
            fastlid.set_languages = target_languages
    elif module == "langid":
        import langid

        classifier = langid.classify
        if target_languages != None:
            target_languages = [
                lang for lang in target_languages if lang in langid_languages
            ]
            langid.set_languages(target_languages)
    else:
        raise ValueError(f"Wrong module {module}")

    lang = classifier(text)[0]

    return lang

def split_alpha_nonalpha(text):
    return re.split(
        r"(?:(?<=[\u4e00-\u9fff])|(?<=[\u3040-\u30FF]))(?=[a-zA-Z])|(?<=[a-zA-Z])(?:(?=[\u4e00-\u9fff])|(?=[\u3040-\u30FF]))",
        text,
    )
    
def check_is_none(item) -> bool:
    """none -> True, not none -> False"""
    return (
        item is None
        or (isinstance(item, str) and str(item).isspace())
        or str(item) == ""
    )
