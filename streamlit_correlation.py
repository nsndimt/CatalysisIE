import pandas as pd
import streamlit as st

from collections import Counter, defaultdict

@st.cache(allow_output_mutation=True)
def load_pred_label():
    return pd.read_json('index/pred_label_data.json').set_index(['doi', 'sent_i']).sort_index()

if __name__ == "__main__":
    try:
        st.commands.page_config.set_page_config(page_title='Catalysis Correlation Analysis',layout='wide')
    except:
        pass
    pred_label_data = load_pred_label()
    pred_label_data_index = set(pred_label_data.index.to_list())

    icon_cols = st.columns([2, 1, 3, 1])
    with icon_cols[0]:
        st.image('icon.png')
    with icon_cols[2]:
        st.markdown(f'## Catalysis Correlation Analysis')

    left, right = st.columns([7,3])
    with left:
        text = st.text_input('Input your keywords here(e.g. Ru):')
    with right:
        option = st.selectbox('Type of keywords?',
                          ('Catalyst', 'Reactant', 'Product', 'Reaction', 'Characterization', 'Treatment'))
    # with right:
    #     values = st.slider('Select a range of context', 0, 20)

    if text:
        selected = pred_label_data[pred_label_data.label == option]

        span_counter = defaultdict(lambda: Counter())
        span_mapping = defaultdict(lambda: defaultdict(lambda: Counter()))
        collect_sent_index = set()
        for sent in selected[selected.norm_span.str.contains(text, case=False)].itertuples():
            doi, sent_i = sent.Index
            collect_sent_index.add((doi, sent_i))
            # for i in range(values):
            #     collect_sent_index.add((doi, sent_i-i-1))
            # for i in range(values):
            #     collect_sent_index.add((doi, sent_i+i+1))

        collect_sent_index = collect_sent_index & pred_label_data_index
        if len(collect_sent_index) == 0:
            st.markdown('##### No result found')

        for span in pred_label_data.loc[list(collect_sent_index)].itertuples():
            if len(span.norm_span) == 0:
                continue
            span_counter[span.label][span.norm_span] += 1
            span_mapping[span.label][span.norm_span][span.span]+=1
    else:
        st.markdown('##### No keywords specified, show statistics on the whole corpus')
        span_counter = defaultdict(lambda: Counter())
        span_mapping = defaultdict(lambda: defaultdict(lambda: Counter()))
        for span in pred_label_data.itertuples():
            if len(span.norm_span) == 0:
                continue
            span_counter[span.label][span.norm_span] += 1
            span_mapping[span.label][span.norm_span][span.span]+=1

    label_names = ['Catalyst', 'Reactant', 'Product', 'Reaction', 'Characterization', 'Treatment']
    span_counter_mapping = [(l, span_counter[l], span_mapping[l]) for l in label_names if len(span_counter[l])>0]

    def draw_label_df(label, span_c, span_m):
        st.markdown(f'### {label}s')
        lines = []
        for n, nf in span_c.most_common():
            raw = []
            for r, rf in span_m[n].most_common(10):
                raw.append(f'{r}({rf})')
            lines.append((n, ';'.join(raw), nf))
        df = pd.DataFrame(lines, columns=['Name', 'Raw', 'Freq'], index=[i + 1 for i in range(len(span_c))])
        df = df.set_index('Name')
        df.index.name = 'Name'
        st.dataframe(df, width=500)

    while len(span_counter_mapping) > 0:
        for col in st.columns(min(len(span_counter_mapping),2)):
            label, span_c, span_m = span_counter_mapping.pop(0)
            with col:
                draw_label_df(label, span_c, span_m)

    head_left, head_mid, head_right = st.columns(3)
    with head_mid:
        st.markdown(f'#### Jump to [Search Engine](http://128.4.30.138:8080/)')
