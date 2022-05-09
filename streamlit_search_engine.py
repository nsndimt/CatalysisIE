import pymongo
import streamlit as st
import time
import json
from pyserini.search import SimpleSearcher
import stanza
from collections import defaultdict, Counter
from utils import get_bio_spans


@st.cache(allow_output_mutation=True)
def load_db():
    client = pymongo.MongoClient('infochain.ece.udel.edu', 27018)
    parse_col = client["elsevier"]["parse"]
    return parse_col

@st.cache(allow_output_mutation=True)
def fetch_paper(doi):
    global db
    res = db.find_one({'DOI': doi})
    assert res is not None, 'not find doi in crawl'
    return res


@st.cache(allow_output_mutation=True)
def load_tokenizer():
    print('running load tokenizer')
    nlp = stanza.Pipeline('en', package='craft', processors='tokenize', use_gpu=False)
    return nlp


@st.cache(allow_output_mutation=True, hash_funcs={stanza.pipeline.core.Pipeline:lambda _:None})
def tokenize(txt):
    global nlp
    #force single sentence
    sent_token = []
    idx = 0
    for sent in nlp(txt).sentences:
        for token in sent.tokens:
            assert len(token.id) == 1, 'multi word token'
            sent_token.append({
                'text': token.text,
                "id": idx,
                "start": token.start_char,
                "end": token.end_char,
            })
            idx += 1
    return sent_token


def render_para(tokenized_query, para_sents):
    qtext = [token["text"] for token in tokenized_query]

    spans = []
    para_text = []
    for sent_i, sent in para_sents:
        prev_end = None
        prev_tag = 'O'
        prev_type = None
        begin_offset = None
        for token in sent:
            token_tag, token_type = token['pred'].split('-') if token['pred'] != 'O' else ('O', None)
            if (prev_tag, token_tag) in [('B', 'B'), ('B', 'O'), ('I', 'B'), ('I', 'O')]:
                spans.append((begin_offset, len(para_text) - 1, prev_type))

            prev_tag = token_tag
            prev_type = token_type

            if prev_end and token['start'] > prev_end:
                para_text.append(' ')
                para_text.append(token["text"])
            else:
                para_text.append(token["text"])

            if token_tag == 'B':
                begin_offset = len(para_text) - 1

            prev_end = token['end']
        if (prev_tag, 'O') in [('B', 'B'), ('B', 'O'), ('I', 'B'), ('I', 'O')]:
            spans.append((begin_offset, len(para_text) - 1, prev_type))
        para_text.append('\n')

    text_highlight = []
    for t in para_text:
        if not t.isalnum() or t not in qtext:
            text_highlight.append(t)
        else:
            text_highlight.append(f'**`{t}`**')

    colors = {'Catalyst': [130, 224, 170], 'Reactant': [244, 208, 63], 'Product': [93, 173, 226],
              'Reaction': [153, 168, 50], 'Characterization': [165, 105, 189], 'Treatment': [40, 51, 204]}
    start2color = {i:colors[l] for i, j ,l in spans}
    end2color = {j:colors[l] for i, j ,l in spans}

    text_highlight_colored = []
    for i,t in enumerate(text_highlight):
        if i in end2color and i in start2color:
            r, g, b = start2color[i]
            text_highlight_colored.append(f'<span style="color:rgb({r}, {g}, {b})">{t}</span>')
        elif i in start2color:
            r,g,b = start2color[i]
            text_highlight_colored.append(f'<span style="color:rgb({r}, {g}, {b})">{t}')
        elif i in end2color:
            text_highlight_colored.append(f'{t}</span>')
        else:
            text_highlight_colored.append(t)
    text_markdown = ''.join(text_highlight_colored)
    return text_markdown


def render_paper(tokenized_query, doc):
    global para_topk
    doi, para_buf = doc
    paper = fetch_paper(doi)
    st.markdown(f"## [{paper['Title']}](https://doi.org/{paper['DOI']})")
    if len(paper['Journal']) > 0:
        st.markdown(f"#### {paper['Journal']}")
    # if len(paper['Keywords']) > 0:
    #     st.markdown(f"#### Keywords: {' ,'.join(paper['Keywords'])}")

    with st.expander("Highlighted Paragraphs"):
        for d, para_i, para_sents in para_buf[:para_topk]:
            text_markdown = render_para(tokenized_query, para_sents)
            st.markdown(f'+ {text_markdown}', unsafe_allow_html=True)


def collect_spans(res):
    label_names = ['Catalyst', 'Reactant', 'Product', 'Reaction', 'Characterization', 'Treatment']
    spans = {l:Counter() for l in label_names}
    para2spans = defaultdict(list)

    for d, idx in res:
        doi, para_i, para_sents = idx
        for sent_i, sent in para_sents:
            sent_tags = [token.get('pred', 'O') for token in sent]
            for start, end, label in get_bio_spans(sent_tags):
                span_text = ''
                prev_end = None
                for token in sent[start:end + 1]:
                    if prev_end and token['start'] > prev_end:
                        span_text += f' {token["text"]}'
                    else:
                        span_text += token['text']
                    prev_end = token['end']
                spans[label][span_text] += 1
                para2spans[(doi, para_i)].append(span_text)
    return spans, para2spans


def render_span_filter(spans):
    select_span = set()
    for l, l_spans in spans.items():
        if len(l_spans) == 0:
            continue
        else:
            st.sidebar.markdown(f'## {l}')
            checkbox2text = [t for t, n in l_spans.most_common(20)]

            def clear_page():
                st.session_state.page = 0

            selected = [st.sidebar.checkbox(f'{t}({n})', on_change=clear_page) for t, n in l_spans.most_common(20)]
            if any(selected):
                for text, checked in zip(checkbox2text, selected):
                    if checked:
                        select_span.add(text)
    return select_span


def filter_span(res, select_span, para2spans):
    if len(select_span) > 0:
        select_res = []
        for d, idx in res:
            doi, para_i, para_sents = idx
            find = False
            for span_text in para2spans[(doi, para_i)]:
                if any([sst in span_text for sst in select_span]):
                    find = True
                    break
            if find:
                select_res.append((d, idx))
        return select_res
    else:
        return res


def display(query, doc_ranked):
    tokenized_query = tokenize(query)
    for card in doc_ranked:
        render_paper(tokenized_query, card)


@st.cache(allow_output_mutation=True)
def load_sparse_index():
    print('load sparse index')
    searcher = SimpleSearcher('index/para_w10_s10')
    searcher.set_bm25(k1=1.25, b=0.9)
    return searcher


@st.cache(allow_output_mutation=True)
def sparse_search(target_text, k=1000):
    print('run sparse search')
    global searcher
    if isinstance(target_text, str):
        target_text = [target_text]

    res = []
    qids = [f'qid#{i}' for i in range(len(target_text))]
    hits = searcher.batch_search(target_text, qids, k, threads=4)
    for qid in qids:
        qid_hits = hits[qid]
        qid_res = []
        for i in range(len(qid_hits)):
            d = qid_hits[i].score
            doi, para_i = qid_hits[i].docid.split('#')
            para_sents = json.loads(qid_hits[i].raw)['raw_json']
            qid_res.append((d, (doi, para_i, para_sents)))
        res.append(qid_res)

    if len(res) == 1:
        res = res[0]
    return res


def paper_rank(res):
    global para_topk
    paper_group = defaultdict(list)
    for d, idx in res:
        doi, para_i, para_sents = idx
        paper_group[doi].append((d, para_i, para_sents))

    def sum_top_para_score(item):
        doi, paras = item
        paras = sorted(paras, key=lambda x: x[0], reverse=True)
        return sum([x[0] for x in paras[:para_topk]])

    return sorted(list(paper_group.items()), key=sum_top_para_score, reverse=True)


def paging(doc_ranked):
    global doc_topk

    if len(doc_ranked) > doc_topk:
        if 'page' not in st.session_state:
            st.session_state.page = 0
            cur_page = 0
        else:
            cur_page = st.session_state.page

        doc_in_page = doc_ranked[cur_page*doc_topk:cur_page*doc_topk+doc_topk]

        def render_page_control():
            def prev_page():
                st.session_state.page = max(0, st.session_state.page - 1)

            def next_page():
                max_page = (len(doc_ranked)+doc_topk-1)//doc_topk
                st.session_state.page = min(max_page, st.session_state.page + 1)

            left, mid, right = st.columns(3)
            with left:
                st.button('Prev', on_click=prev_page)
            with mid:
                st.markdown(f'### Page No.{cur_page+1}')
            with right:
                st.button('Next', on_click=next_page)
        return doc_in_page, render_page_control
    else:
        def render_page_control():
            pass
        return doc_ranked, render_page_control


def search(qtext):
    start = time.time()
    sparse_res = sparse_search(qtext, k=1000)
    end = time.time()
    print('sparse retrieval latency', end - start)
    start = time.time()
    spans, para2spans = collect_spans(sparse_res)
    end = time.time()
    print('collect latency', end - start)
    start = time.time()
    select_spans = render_span_filter(spans)
    end = time.time()
    print('render condition latency', end - start)
    start = time.time()
    filtered_res = filter_span(sparse_res,select_spans,para2spans)
    end = time.time()
    print('filter para latency', end - start)
    start = time.time()
    doc_ranked = paper_rank(filtered_res)
    doc_in_page, render_page_control = paging(doc_ranked)
    end = time.time()
    print('rank doc latency', end - start)
    start = time.time()
    display(qtext, doc_in_page)
    render_page_control()
    end = time.time()
    print('render doc latency', end - start)


def query():
    icon_cols = st.columns([2,1,3,1])
    with icon_cols[0]:
        st.image('icon.png')
    with icon_cols[2]:
        st.markdown(f'## Catalysis Search Engine')
    qtext = st.text_input('Input your criterion here(e.g. Ru/TiO2):')
    return qtext


if __name__ == "__main__":
    try:
        st.commands.page_config.set_page_config(page_title='Catalysis Search Engine',layout='wide')
    except:
        pass
    db = load_db()
    searcher = load_sparse_index()
    nlp = load_tokenizer()
    # model = load_fasttext()
    doc_topk = 10
    para_topk = 3
    qtext = query()
    search(qtext)
