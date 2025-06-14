import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import plotly.graph_objects as go

# ======== DATA CLEANING FUNCTION ========
def clean_dataset(df):
    """Automatically fixes missing values and anomalies in data"""
    original_stats = {
        'rows': len(df),
        'nulls': df.isnull().sum().sum(),
        'duplicates': df.duplicated().sum()
    }

    # 1. Handle missing values
    for col in df.columns:
        # For categorical features
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('Unknown').str.strip().str.capitalize()

        # For numeric features
        elif pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())

    # 2. Fix date columns
    date_cols = [col for col in df.columns if 'date' in col.lower() or 'ts' in col.lower()]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce').fillna(pd.to_datetime('today'))

    # 3. Remove duplicates
    df = df.drop_duplicates()

    new_stats = {
        'rows': len(df),
        'nulls': df.isnull().sum().sum(),
        'duplicates': df.duplicated().sum()
    }

    return df, original_stats, new_stats

# Load data with caching
@st.cache_data
def load_and_clean_data():
    events = pd.read_csv("events-2.csv")
    clients = pd.read_excel("clients.xlsx")
    subscriptions = pd.read_csv("subscriptions.csv")

    events, events_orig, events_new = clean_dataset(events)
    clients, clients_orig, clients_new = clean_dataset(clients)
    subscriptions, subs_orig, subs_new = clean_dataset(subscriptions)

    return (events, clients, subscriptions), (events_orig, clients_orig, subs_orig), (events_new, clients_new, subs_new)

# Load data
(data_fixed, original_stats, fixed_stats) = load_and_clean_data()
events_fixed, clients_fixed, subs_fixed = data_fixed

# Filtering function
def apply_filters(events, clients, subscriptions, segment_filter, sector_filter, plan_filter, status_filter):
    filtered_clients = clients.copy()

    if segment_filter:
        filtered_clients = filtered_clients[filtered_clients['segment'].isin(segment_filter)]
    if sector_filter:
        filtered_clients = filtered_clients[filtered_clients['sector'].isin(sector_filter)]

    filtered_subs = subscriptions[subscriptions['client_id'].isin(filtered_clients['client_id'])]
    if plan_filter:
        filtered_subs = filtered_subs[filtered_subs['plan'].isin(plan_filter)]
    if status_filter:
        filtered_subs = filtered_subs[filtered_subs['status'].isin(status_filter)]

    filtered_events = events[events['client_id'].isin(filtered_clients['client_id'])]

    return filtered_events, filtered_clients, filtered_subs

# Data preprocessing
@st.cache_data
def preprocess_data(_events, _clients, _subscriptions):
    _subscriptions = _subscriptions.copy()
    _clients = _clients.copy()
    _events = _events.copy()

    _subscriptions['duration_days'] = (_subscriptions['end_date'] - _subscriptions['start_date']).dt.days
    _clients['registration_month'] = _clients['registration_date'].dt.to_period('M').dt.to_timestamp()
    _events['event_month'] = _events['event_ts'].dt.to_period('M').dt.to_timestamp()

    return _events, _clients, _subscriptions

# Apply preprocessing
events, clients, subscriptions = preprocess_data(events_fixed, clients_fixed, subs_fixed)

# ======== SIDEBAR FILTERS ========
st.sidebar.header("Data Filters")
segment_filter = st.sidebar.multiselect(
    "Customer Segment",
    options=clients['segment'].dropna().unique(),
    default=None
)
sector_filter = st.sidebar.multiselect(
    "Industry Sector",
    options=clients['sector'].dropna().unique(),
    default=None
)
plan_filter = st.sidebar.multiselect(
    "Subscription Plan",
    options=subscriptions['plan'].dropna().unique(),
    default=None
)
status_filter = st.sidebar.multiselect(
    "Subscription Status",
    options=subscriptions['status'].dropna().unique(),
    default=None
)

# Apply filters
filtered_events, filtered_clients, filtered_subs = apply_filters(
    events, clients, subscriptions,
    segment_filter, sector_filter,
    plan_filter, status_filter
)

# Merge filtered data
merged = filtered_clients.merge(
    filtered_subs,
    on='client_id',
    how='left'
).merge(
    filtered_events.groupby('client_id')['event_type'].count().reset_index(name='event_count'),
    on='client_id',
    how='left'
)

# ======== DASHBOARD ========
st.title("📊 Customer Analytics Dashboard")

# ======== DATA QUALITY SECTION ========
st.header("🛠️ Data Quality Report")

# 1. Comparison table
st.subheader("Data Cleaning Summary")
comparison = pd.DataFrame({
    'Dataset': ['Events', 'Customers', 'Subscriptions'],
    'Original Rows': [s['rows'] for s in original_stats],
    'Cleaned Rows': [s['rows'] for s in fixed_stats],
    'Duplicates Removed': [o['duplicates'] for o in original_stats],
    'Nulls Fixed': [o['nulls'] for o in original_stats]
})

st.dataframe(comparison.style.format({
    'Original Rows': '{:,}',
    'Cleaned Rows': '{:,}',
    'Duplicates Removed': '{:,}',
    'Nulls Fixed': '{:,}'
}))

# 2. Detailed reports
st.subheader("Detailed Dataset Reports")

tabs = st.tabs(["Events", "Customers", "Subscriptions"])
for i, (name, df) in enumerate(zip(["Events", "Customers", "Subscriptions"], [events_fixed, clients_fixed, subs_fixed])):
    with tabs[i]:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Data Types:**")
            st.write(df.dtypes.to_frame('Type'))
        with col2:
            st.markdown("**Missing Values:**")
            st.write(df.isnull().sum().to_frame('Count'))
        st.markdown("**Data Sample (5 rows):**")
        st.dataframe(df.head())

# 3. Visualization of key changes
st.subheader("🔎 Data Cleaning Visualization")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].bar(['Events', 'Customers', 'Subscriptions'],
          [s['nulls'] for s in original_stats],
          label='Original Nulls')
ax[0].bar(['Events', 'Customers', 'Subscriptions'],
          [s['nulls'] for s in fixed_stats],
          bottom=[s['nulls'] for s in original_stats],
          label='Remaining Nulls')
ax[0].set_title("Null Values Handling")
ax[0].legend()

ax[1].bar(['Events', 'Customers', 'Subscriptions'],
          [s['duplicates'] for s in original_stats])
ax[1].set_title("Duplicate Records Removed")

st.pyplot(fig)
st.success("Data processing complete! Proceed with analysis.")

# ======== LTV ANALYSIS ========
st.header("💸 Customer Lifetime Value (LTV)")

# Methodology explanation
st.markdown("""
**Calculation Methodology:**  
1. Sum all customer payments  
2. Consider customer lifetime  
3. Compare across segments  
4. Identify key patterns  
""")

# Data preparation
try:
    subs = filtered_subs.copy()
    clients = filtered_clients.copy()

    subs['amount'] = pd.to_numeric(subs['amount'], errors='coerce')
    clients['registration_date'] = pd.to_datetime(clients['registration_date'], errors='coerce')

    ltv_data = (
        subs.groupby('client_id')
        .agg(
            total_revenue=('amount', 'sum'),
            subscriptions=('subscription_id', 'count')
        )
        .merge(
            clients[['client_id', 'segment', 'registration_date']],
            on='client_id',
            how='left'
        )
        .dropna(subset=['total_revenue', 'registration_date'])
        .assign(
            segment=lambda x: x['segment'].fillna('Unknown'),
            lifetime_days=lambda x: (pd.to_datetime('today') - x['registration_date']).dt.days,
            ltv_per_day=lambda x: x['total_revenue'] / x['lifetime_days']
        )
        .query("lifetime_days > 0 and total_revenue > 0")
    )

except Exception as e:
    st.error(f"Data preparation error: {str(e)}")
    st.stop()

# Visualization
st.subheader("📊 LTV Visualization")

fig = plt.figure(figsize=(18, 12))
gs = gridspec.GridSpec(2, 2, figure=fig)

# Plot 1: LTV Distribution
ax1 = fig.add_subplot(gs[0, 0])
sns.boxplot(
    data=ltv_data,
    x='segment',
    y='total_revenue',
    ax=ax1,
    palette='Blues'
)
ax1.set_title("LTV Distribution by Segment")
ax1.set_ylabel("Total Revenue ($)")
ax1.ticklabel_format(style='plain', axis='y')

# Plot 2: Top Customers
ax2 = fig.add_subplot(gs[0, 1])
top_clients = ltv_data.nlargest(10, 'total_revenue')
sns.barplot(
    data=top_clients,
    x='total_revenue',
    y='client_id',
    hue='segment',
    dodge=False,
    ax=ax2,
    palette='viridis'
)
ax2.set_title("Top 10 Customers by LTV")
ax2.set_xlabel("Total Revenue ($)")
ax2.set_ylabel("Customer ID")

# Plot 3: LTV vs Lifetime
ax3 = fig.add_subplot(gs[1, :])
sns.scatterplot(
    data=ltv_data,
    x='lifetime_days',
    y='total_revenue',
    hue='segment',
    size='subscriptions',
    sizes=(20, 200),
    alpha=0.7,
    ax=ax3,
    palette='Set2'
)
ax3.set_title("LTV vs Customer Lifetime")
ax3.set_xlabel("Days Since Registration")
ax3.set_ylabel("LTV ($)")

plt.tight_layout()
st.pyplot(fig)

# Key Metrics
st.subheader("🔢 Key Metrics")

if not ltv_data.empty:
    cols = st.columns(4)
    with cols[0]:
        st.metric("Total LTV", f"{ltv_data['total_revenue'].sum():,.0f} $")
    with cols[1]:
        st.metric("Average LTV", f"{ltv_data['total_revenue'].mean():,.0f} $")
    with cols[2]:
        st.metric("Median LTV", f"{ltv_data['total_revenue'].median():,.0f} $")
    with cols[3]:
        st.metric("Max LTV", f"{ltv_data['total_revenue'].max():,.0f} $")

# Insights
if not ltv_data.empty:
    st.subheader("💡 Insights")
    top_segment = ltv_data.groupby('segment')['total_revenue'].mean().idxmax()
    revenue_concentration = top_clients['total_revenue'].sum() / ltv_data['total_revenue'].sum()

    st.markdown(f"""
    1. Highest value segment: **{top_segment}**  
    2. Top 10 customers contribute **{revenue_concentration:.1%}** of total revenue  
    3. Average customer generates **{ltv_data['ltv_per_day'].mean():.2f} $/day**  
    """)

# Detailed Data
if st.checkbox("Show detailed LTV data"):
    st.dataframe(
        ltv_data.sort_values('total_revenue', ascending=False)
        .style
        .format({
            'total_revenue': '{:,.0f} $',
            'lifetime_days': '{:.0f}',
            'ltv_per_day': '{:.2f} $/day'
        })
        .background_gradient(subset=['total_revenue'], cmap='Blues')
    )

# ======== REPEAT SUBSCRIPTIONS ========
st.header("🔄 Subscription Renewal Analysis")

# Methodology
st.markdown("""
**Analysis Focus:**  
- Subscription renewal frequency  
- Loyal customer percentage  
- Revenue impact  
""")

# Data preparation
try:
    subs_count = filtered_subs.groupby('client_id').agg(
        subscriptions=('subscription_id', 'count'),
        total_amount=('amount', 'sum')
    ).reset_index()

    subs_count = subs_count.merge(
        filtered_clients[['client_id', 'segment']],
        on='client_id',
        how='left'
    ).fillna({'segment': 'Unknown'})

    subs_count['client_type'] = pd.cut(
        subs_count['subscriptions'],
        bins=[0, 1, 2, 5, float('inf')],
        labels=['One-time', 'Repeat (2)', 'Regular (3-5)', 'VIP (5+)']
    )

except Exception as e:
    st.error(f"Data processing error: {str(e)}")
    st.stop()

# Key Metrics
st.subheader("📊 Key Metrics")

if not subs_count.empty:
    repeat_share = (subs_count['subscriptions'] > 1).mean()
    vip_share = (subs_count['subscriptions'] > 5).mean()

    col1, col2, col3 = st.columns(3)
    col1.metric(
        label="Repeat Customers",
        value=f"{repeat_share:.1%}",
        help="Customers with 2+ subscriptions"
    )
    col2.metric(
        label="VIP Customers (5+)",
        value=f"{vip_share:.1%}",
        help="Most loyal customers"
    )
    col3.metric(
        label="Revenue from Renewals",
        value=f"{subs_count[subs_count['subscriptions'] > 1]['total_amount'].sum()/subs_count['total_amount'].sum():.1%}",
        help="Percentage of total revenue"
    )

# Visualization
st.subheader("📈 Customer Distribution")

if not subs_count.empty:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    type_dist = subs_count['client_type'].value_counts(normalize=True)
    type_dist.plot.pie(
        ax=ax1,
        autopct='%1.1f%%',
        startangle=90,
        colors=['#ff9999','#66b3ff','#99ff99','#ffcc99'],
        wedgeprops=dict(width=0.3)
    )
    ax1.set_title("Customer Types by Subscription Count")
    ax1.set_ylabel("")

    sns.barplot(
        data=subs_count,
        x='client_type',
        y='total_amount',
        estimator='mean',
        ax=ax2,
        palette='Blues_d'
    )
    ax2.set_title("Average Revenue by Customer Type")
    ax2.set_xlabel("Customer Type")
    ax2.set_ylabel("Average Revenue ($)")
    ax2.ticklabel_format(style='plain', axis='y')

    plt.tight_layout()
    st.pyplot(fig)

# Segment Analysis
st.subheader("🔍 Segment Analysis")

if not subs_count.empty:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Renewal Rate by Segment:**")
        segment_repeat = subs_count.groupby('segment')['subscriptions'].apply(
            lambda x: (x > 1).mean()
        ).sort_values(ascending=False)
        st.dataframe(
            segment_repeat.reset_index(name='Rate').style.format({'Rate': '{:.1%}'}),
            height=300
        )

    with col2:
        fig = plt.figure(figsize=(8, 4))
        sns.boxplot(
            data=subs_count,
            x='segment',
            y='total_amount',
            palette='Set2'
        )
        plt.title("Revenue Distribution by Segment")
        plt.xticks(rotation=45)
        st.pyplot(fig)

# Recommendations
st.subheader("💡 Recommendations")

if not subs_count.empty:
    best_segment = subs_count.groupby('segment')['subscriptions'].mean().idxmax()
    best_repeat_rate = (subs_count[subs_count['segment'] == best_segment]['subscriptions'] > 1).mean()

    st.markdown(f"""
1. **For {best_segment} segment** (renewal rate: {best_repeat_rate:.1%}):
   - Loyalty programs
   - Exclusive renewal offers

2. **For one-time customers**:
   - Special renewal incentives
   - Churn analysis

3. **For VIP customers**:
   - Dedicated account managers
   - Additional benefits
    """)

# ======== ACTIVITY ANALYSIS ========
st.header("📊 Customer Activity")

st.metric(
    "Average Activity",
    f"{merged['event_count'].mean():.1f} events",
    help="Average events per customer"
)

# Activity visualization
fig1 = plt.figure(figsize=(12, 6))
segment_stats = merged.groupby('segment')['event_count'].agg(['mean', 'median']).reset_index()
segment_stats = segment_stats.sort_values('mean', ascending=False)

sns.barplot(
    data=segment_stats,
    x='segment',
    y='mean',
    color='skyblue',
    label='Mean'
)

sns.scatterplot(
    data=segment_stats,
    x='segment',
    y='median',
    color='red',
    s=100,
    label='Median'
)

plt.title("Activity by Customer Segment", pad=15)
plt.xlabel("Customer Segment")
plt.ylabel("Event Count")
plt.xticks(rotation=45)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.3)

st.pyplot(plt.gcf())

# ======== CUSTOMER LIFECYCLE ========
st.header("📈 Customer Lifecycle Analysis")

try:
    clients_subs = filtered_subs.merge(
        filtered_clients[['client_id', 'registration_date']],
        on='client_id',
        how='left'
    )

    clients_subs['days_to_subscribe'] = (clients_subs['start_date'] - clients_subs['registration_date']).dt.days

    if 'client_id' in filtered_events.columns and 'event_ts' in filtered_events.columns:
        last_event = filtered_events.groupby('client_id')['event_ts'].max().reset_index(name='last_event')
        lifecycle_df = clients_subs.merge(last_event, on='client_id', how='left')

        if 'start_date' in lifecycle_df.columns and 'last_event' in lifecycle_df.columns:
            lifecycle_df['active_days_after_sub'] = (lifecycle_df['last_event'] - lifecycle_df['start_date']).dt.days
        else:
            lifecycle_df['active_days_after_sub'] = None
    else:
        lifecycle_df = clients_subs.copy()
        lifecycle_df['active_days_after_sub'] = None

    if 'segment' in filtered_clients.columns:
        lifecycle_df = lifecycle_df.merge(
            filtered_clients[['client_id', 'segment']],
            on='client_id',
            how='left'
        )

    if not lifecycle_df.empty:
        sub_col1, sub_col2, sub_col3 = st.columns(3)

        with sub_col1:
            convert_days = lifecycle_df['days_to_subscribe'].median() if 'days_to_subscribe' in lifecycle_df.columns else None
            st.metric(
                "Median Time to Subscribe",
                f"{convert_days:.0f} days" if convert_days is not None else "N/A"
            )

        with sub_col2:
            active_days = lifecycle_df['active_days_after_sub'].median() if 'active_days_after_sub' in lifecycle_df.columns else None
            st.metric(
                "Active Period",
                f"{active_days:.0f} days" if active_days is not None else "N/A"
            )

        with sub_col3:
            churn = (lifecycle_df['active_days_after_sub'] < 30).mean() if 'active_days_after_sub' in lifecycle_df.columns else None
            st.metric(
                "30-Day Churn Rate",
                f"{churn:.1%}" if churn is not None else "N/A"
            )

        if all(col in lifecycle_df.columns for col in ['days_to_subscribe', 'active_days_after_sub']):
            plt.figure(figsize=(12, 8))
            scatter = sns.scatterplot(
                data=lifecycle_df,
                x='days_to_subscribe',
                y='active_days_after_sub',
                hue='plan' if 'plan' in lifecycle_df.columns else None,
                size='active_days_after_sub',
                sizes=(50, 300),
                alpha=0.7,
                palette='viridis'
            )

            plt.axvline(lifecycle_df['days_to_subscribe'].median(),
                        color='red',
                        linestyle='--',
                        linewidth=1,
                        label=f'Median: {lifecycle_df["days_to_subscribe"].median():.0f} days')

            plt.axhline(lifecycle_df['active_days_after_sub'].median(),
                        color='blue',
                        linestyle='--',
                        linewidth=1,
                        label=f'Median: {lifecycle_df["active_days_after_sub"].median():.0f} days')

            plt.title("Customer Lifecycle Analysis\n(Days to Subscribe vs Post-Subscription Activity)")
            plt.xlabel("Days from Registration to Subscription")
            plt.ylabel("Days Active After Subscription")
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            if not lifecycle_df.empty:
                top_active = lifecycle_df.nlargest(5, 'active_days_after_sub')
                for _, row in top_active.iterrows():
                    plt.annotate(f"ID: {row['client_id']}",
                                 (row['days_to_subscribe'], row['active_days_after_sub']),
                                 textcoords="offset points",
                                 xytext=(0,10),
                                 ha='center',
                                 fontsize=8)

            st.pyplot(plt.gcf())

        st.subheader("💡 Key Insights")
        insights = []

        if 'days_to_subscribe' in lifecycle_df.columns:
            fast_convert = lifecycle_df[lifecycle_df['days_to_subscribe'] < 3]
            if not fast_convert.empty and 'plan' in fast_convert.columns:
                best_plan = fast_convert['plan'].mode()[0]
                insights.append(f"• Fast-converting customers (<3 days) prefer **{best_plan}** plan")

        if 'active_days_after_sub' in lifecycle_df.columns:
            long_active = lifecycle_df[lifecycle_df['active_days_after_sub'] > 90]
            if not long_active.empty and 'segment' in long_active.columns:
                active_segment = long_active['segment'].mode()[0]
                insights.append(f"• Most active customers (90+ days) typically from **{active_segment}** segment")

        if insights:
            st.markdown("\n".join(insights))

except Exception as e:
    st.error(f"Lifecycle analysis error: {str(e)}")

# ======== EXPIRED SUBSCRIPTION ACTIVITY ========
st.header("📍 Post-Subscription Activity Analysis")

def analyze_expired_activity(events, subs):
    last_subs = subs.sort_values(['client_id', 'end_date']).groupby('client_id').last().reset_index()
    expired_subs = last_subs[last_subs['status'] == 'expired']
    expired_clients_events = events[events['client_id'].isin(expired_subs['client_id'])]

    active_after_expired = []
    for _, sub in expired_subs.iterrows():
        client_events = expired_clients_events[expired_clients_events['client_id'] == sub['client_id']]
        events_after_end = client_events[client_events['event_ts'] > sub['end_date']]

        if not events_after_end.empty:
            active_after_expired.append({
                'client_id': sub['client_id'],
                'days_since_expired': (events_after_end['event_ts'].max() - sub['end_date']).days,
                'event_count': len(events_after_end),
                'last_plan': sub['plan']
            })

    return pd.DataFrame(active_after_expired)

try:
    expired_activity = analyze_expired_activity(filtered_events, filtered_subs)

    col1, col2, col3 = st.columns(3)
    with col1:
        ratio = len(expired_activity) / len(filtered_clients['client_id'].unique()) if len(filtered_clients) > 0 else 0
        st.metric(
            "Active Without Subscription",
            f"{ratio:.1%}",
            help="Customers active after subscription expired"
        )
    with col2:
        avg_days = expired_activity['days_since_expired'].mean() if not expired_activity.empty else 0
        st.metric(
            "Average Active Days",
            f"{avg_days:.0f}",
            help="Average days of post-expiry activity"
        )
    with col3:
        avg_events = expired_activity['event_count'].mean() if not expired_activity.empty else 0
        st.metric(
            "Average Events",
            f"{avg_events:.1f}",
            help="Average events after expiry"
        )

    if not expired_activity.empty:
        st.subheader("🔍 Activity Details")
        st.write("#### Most Active Without Subscription")
        st.dataframe(
            expired_activity.sort_values('days_since_expired', ascending=False)
            .head(10)
            .style.format({
                'days_since_expired': '{:.0f}',
                'event_count': '{:.0f}'
            })
        )

        fig1, ax1 = plt.subplots(figsize=(8, 4))
        expired_activity['last_plan'].value_counts().plot(kind='bar', ax=ax1)
        ax1.set_title("Last Plan Distribution")
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots(figsize=(10, 4))
        sns.histplot(expired_activity['days_since_expired'], bins=30, kde=True, ax=ax2)
        ax2.set_title("Post-Expiry Activity Duration")
        ax2.set_xlabel("Days After Subscription End")
        st.pyplot(fig2)

    st.subheader("💡 Business Implications")
    insights = []
    if not expired_activity.empty:
        most_common_plan = expired_activity['last_plan'].mode()[0]
        avg_days_active = expired_activity['days_since_expired'].mean()

        insights.extend([
            f"**• {ratio:.1%} customers** continue using platform without active subscription",
            f"**• Mostly from '{most_common_plan}' plan**",
            f"**• Average {avg_days_active:.0f} days** post-expiry activity",
            "",
            "**Recommendations:**",
            "1. Introduce trial period for these customers",
            "2. Personalized renewal offers",
            "3. Analyze if features are available without payment"
        ])
    else:
        insights.append("No post-expiry activity detected - good access control")

    st.markdown("\n".join(insights))

except Exception as e:
    st.error(f"Analysis error: {str(e)}")

# ======== CUSTOMER VALUE ANALYSIS ========
st.header("💎 Customer Value by Segment")

def clean_data(df):
    if 'segment' in df.columns:
        df['segment'] = df['segment'].fillna('Unknown')
    if 'plan' in df.columns:
        df['plan'] = df['plan'].fillna('Unknown')
    if 'amount' in df.columns:
        df['amount'] = df['amount'].fillna(0)
    return df

filtered_clients = clean_data(filtered_clients.copy())
filtered_subs = clean_data(filtered_subs.copy())

try:
    client_metrics = filtered_clients[['client_id', 'segment']].merge(
        filtered_subs[['client_id', 'plan', 'amount']],
        on='client_id',
        how='left'
    )
    client_metrics = clean_data(client_metrics)

    if not client_metrics.empty:
        plan_scores = {
            'Standard': 1,
            'Pro': 2,
            'Enterprise': 3,
            'Unknown': 0
        }

        agg = client_metrics.groupby('segment').agg(
            clients=('client_id', 'nunique'),
            total_subscriptions=('client_id', 'count'),
            total_revenue=('amount', 'sum'),
            avg_revenue_per_client=('amount', lambda x: x.mean() if not x.empty else 0),
            median_revenue=('amount', 'median'),
            revenue_per_sub=('amount', lambda x: x.sum()/x.count()),
            active_ratio=('client_id', lambda x: x.nunique()/filtered_clients['client_id'].nunique()),
            retention_rate=('client_id', lambda x: (x.value_counts() > 1).mean()),
            top_plan=('plan', lambda x: x.mode()[0])
        ).reset_index()

        agg['revenue_share'] = agg['total_revenue'] / agg['total_revenue'].sum()
        agg['growth_potential'] = (agg['avg_revenue_per_client'] / agg['avg_revenue_per_client'].max()) * agg['clients']

        def safe_plan_score(x):
            try:
                return x.map(plan_scores).mean()
            except:
                return 0

        agg['avg_plan_score'] = client_metrics.groupby('segment')['plan'].apply(safe_plan_score)

        metrics_to_rank = ['clients', 'total_subscriptions', 'total_revenue', 'avg_plan_score']
        for metric in metrics_to_rank:
            if metric in agg.columns:
                agg[f'{metric}_rank'] = agg[metric].rank(pct=True)
            else:
                agg[f'{metric}_rank'] = 0

        agg['value_score'] = agg[[f'{m}_rank' for m in metrics_to_rank]].mean(axis=1)

        try:
            agg['segment_priority'] = pd.qcut(
                agg['value_score'],
                3,
                labels=['Low', 'Medium', 'High'],
                duplicates='drop'
            )
        except:
            agg['segment_priority'] = 'Medium'

        st.subheader("📊 Value Analysis Results")
        display_df = agg.copy().sort_values('value_score', ascending=False)
        display_df['segment'] = display_df['segment'].replace('Unknown', 'Unspecified')

        st.dataframe(
            display_df.style
            .background_gradient(subset=['value_score'], cmap='YlGn')
            .format({
                'total_revenue': '{:,.0f} $',
                'avg_revenue_per_client': '{:,.0f} $',
                'avg_plan_score': '{:.1f}',
                'value_score': '{:.2f}'
            }),
            height=500
        )

        st.subheader("📈 Segment Value Visualization")
        fig, ax = plt.subplots(figsize=(10, 6))
        display_df.set_index('segment')['value_score'].sort_values().plot(
            kind='barh',
            color='lightgreen',
            ax=ax
        )
        ax.set_title("Customer Segment Value Index")
        ax.set_xlabel("Value Score")
        st.pyplot(fig)

    else:
        st.warning("⚠️ No data available after processing")

except Exception as e:
    st.error(f"Data processing error: {str(e)}")

# ======== SUBSCRIPTION OVERLAP ANALYSIS ========
st.header("🚨 Subscription Overlap Detection")

st.markdown("""
**Business Implications:**  
Subscription overlap indicates either:  
- System errors  
- Intentional duplication  
- Process issues  
""")

try:
    subs_sorted = filtered_subs.sort_values(['client_id', 'start_date'])
    overlap_cases = []

    for client_id, group in subs_sorted.groupby('client_id'):
        if len(group) < 2:
            continue

        group = group.sort_values('start_date')

        for i in range(len(group)-1):
            current_sub = group.iloc[i]
            next_sub = group.iloc[i+1]

            overlap_start = max(current_sub['start_date'], next_sub['start_date'])
            overlap_end = min(current_sub['end_date'], next_sub['end_date'])

            if overlap_start < overlap_end:
                overlap_days = (overlap_end - overlap_start).days
                overlap_cases.append({
                    'client_id': client_id,
                    'start_date': overlap_start,
                    'duration_days': overlap_days,
                    'plan_1': current_sub['plan'],
                    'plan_2': next_sub['plan']
                })

    if overlap_cases:
        overlap_df = pd.DataFrame(overlap_cases)

        st.subheader("📊 Overlap Statistics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Cases", len(overlap_df))
        col2.metric("Average Duration", f"{overlap_df['duration_days'].mean():.1f} days")
        col3.metric("Unique Customers", overlap_df['client_id'].nunique())

        st.subheader("📈 Monthly Trend")
        monthly_stats = overlap_df.set_index('start_date').resample('M').agg({
            'client_id': 'count',
            'duration_days': 'mean'
        }).rename(columns={'client_id': 'cases'})

        fig1, ax1 = plt.subplots(figsize=(10, 4))
        ax1.bar(monthly_stats.index, monthly_stats['cases'], color='salmon')
        ax1.set_title("Monthly Overlap Cases")
        ax1.set_ylabel("Count")
        plt.xticks(rotation=45)
        st.pyplot(fig1)

        st.subheader("🧩 Plan Combinations")
        st.dataframe(overlap_df.groupby(['plan_1', 'plan_2']).size().reset_index(name='count')
                     .sort_values('count', ascending=False))

        st.subheader("💡 Recommendations")
        st.markdown("""
        - Audit subscription system
        - Analyze frequent overlap cases
        - Consider preventing overlaps
        """)
    else:
        st.success("✅ No overlapping subscriptions detected")

except Exception as e:
    st.error(f"Overlap analysis error: {str(e)}")

# ======== ACTIVITY TRENDS ========
st.header("📈 Customer Activity Trends")

st.markdown("""
**Analysis Focus:**  
- Customer retention patterns  
- Activity intensity changes  
- Event type distribution  
""")

try:
    client_activity = (
        filtered_events.merge(
            filtered_clients[['client_id', 'registration_date', 'segment']],
            on='client_id'
        )
        .assign(
            days_since_reg = lambda x: (x['event_ts'] - x['registration_date']).dt.days,
            month_since_reg = lambda x: x['days_since_reg'] // 30
        )
        .query("month_since_reg >= 0")
    )

    cohort_metrics = client_activity.groupby('month_since_reg').agg(
        unique_clients=('client_id', 'nunique'),
        total_events=('event_id', 'count'),
        events_per_client=('event_id', lambda x: x.count() / x.nunique())
    ).reset_index()

    initial_clients = client_activity[client_activity['month_since_reg'] == 0]['client_id'].nunique()
    cohort_metrics['retention_rate'] = cohort_metrics['unique_clients'] / initial_clients

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    sns.lineplot(data=cohort_metrics, x='month_since_reg', y='unique_clients',
                 marker='o', ax=ax1, color='royalblue')
    ax1.set_title("🆔 Unique Customers")
    ax1.set_xlabel("Months Since Registration")
    ax1.set_ylabel("Customer Count")
    ax1.axhline(initial_clients, color='red', linestyle='--', alpha=0.3)

    sns.lineplot(data=cohort_metrics, x='month_since_reg', y='retention_rate',
                 marker='o', ax=ax2, color='green')
    ax2.set_title("📉 Retention Rate")
    ax2.set_xlabel("Months Since Registration")
    ax2.set_ylabel("Retention %")
    ax2.set_ylim(0, 1)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter('{:.0%}'.format))

    sns.lineplot(data=cohort_metrics, x='month_since_reg', y='events_per_client',
                 marker='o', ax=ax3, color='purple')
    ax3.set_title("⚡ Activity per Customer")
    ax3.set_xlabel("Months Since Registration")
    ax3.set_ylabel("Average Events")

    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("🔍 Event Type Analysis")
    event_analysis = (
        client_activity.groupby(['month_since_reg', 'event_type'])
        ['client_id'].nunique()
        .unstack()
        .fillna(0)
    )

    fig2, ax = plt.subplots(figsize=(10, 5))
    event_analysis.plot(kind='area', stacked=True, ax=ax)
    ax.set_title("Event Type Distribution Over Time")
    ax.set_xlabel("Months Since Registration")
    ax.set_ylabel("Customer Count")
    ax.legend(title='Event Type', bbox_to_anchor=(1.05, 1))
    st.pyplot(fig2)

    st.subheader("💡 Key Insights")
    critical_month = cohort_metrics[cohort_metrics['retention_rate'] < 0.5]['month_since_reg'].min()
    insights = []

    if not pd.isna(critical_month):
        insights.append(f"⚠️ **Critical Point:** >50% churn by month {critical_month}")
    else:
        insights.append("✅ Stable Retention: Over 50% remain active")

    top_event = event_analysis.iloc[-1].idxmax()
    insights.append(f"📌 Dominant Event After 3 Months: {top_event}")

    st.markdown("\n".join(insights))

    st.subheader("🎯 Recommendations")
    st.markdown("""
    1. **0-3 Months:**  
       - Enhance onboarding  
       - Personalized welcome offers  

    2. **Post-Critical Month:**  
       - Retargeting campaigns  
       - Special incentives  

    3. **Active Customers:**  
       - Loyalty programs  
       - Personalized content  
    """)

except Exception as e:
    st.error(f"Trend analysis error: {str(e)}")