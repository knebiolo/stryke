import pandas as pd
from pandas.testing import assert_frame_equal

# Small unit test for route aggregation logic (Policy B) using an in-memory DataFrame

def test_policy_b_aggregation():
    # Construct a small simulation-like DataFrame with state_ and survival_ columns
    data = [
        {'index':0,'iteration':0,'day':'2024-01-01','state_0':'river_node_0','state_1':'Unit A','survival_0':1,'survival_1':1},
        {'index':1,'iteration':0,'day':'2024-01-01','state_0':'river_node_0','state_1':'Unit B','survival_0':1,'survival_1':0},
        {'index':0,'iteration':0,'day':'2024-01-02','state_0':'river_node_0','state_1':'Unit A','survival_0':1,'survival_1':0},
    ]
    df = pd.DataFrame(data)
    # Simulate the relevant part of app.py aggregation
    all_passage_events = []
    for _, row in df.iterrows():
        fish_identifier = row.get('index')
        state_cols = [c for c in df.columns if c.startswith('state_')]
        survival_cols = [c for c in df.columns if c.startswith('survival_')]
        fish_path = [row[col] for col in state_cols if pd.notna(row[col])]
        for pos, state in enumerate(fish_path):
            if 'river_node' not in state.lower():
                survived = None
                if pos>0 and survival_cols and len(survival_cols) >= pos:
                    survived = row.get(survival_cols[pos-1])
                all_passage_events.append({'fish_id':fish_identifier,'iteration':row.get('iteration'),'day':row.get('day'),'passage_route':state,'survived':survived})
                break
    events_df = pd.DataFrame(all_passage_events)
    dedup = events_df.drop_duplicates(subset=['fish_id','iteration','day'])
    # Expect 3 deduped events (each row becomes one event), and counts per route
    counts = dedup['passage_route'].value_counts().to_dict()
    assert counts.get('Unit A') == 2
    assert counts.get('Unit B') == 1
