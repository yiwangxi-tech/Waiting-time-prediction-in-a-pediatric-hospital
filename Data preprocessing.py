import pandas as pd

# Read the Excel file, the 1F throat swab cohort was used as an example
df = pd.read_excel('1F咽拭子.xlsx')

# If it is a radiology examination, replace "分诊签到时间" with "签到/登记时间"
df_rd = df.drop_duplicates(subset=['病人ID', 'Visit times', '分诊签到时间'], keep='first')

# Extract the time information and convert this column to the datetime type. If it is a radiology examination, replace "分诊签到时间" with "签到/登记时间"，replace "采样时间" with "上机检查时间"
df_rd['分诊签到时间'] = pd.to_datetime(df_rd['分诊签到时间'])
df_rd['采样时间'] = pd.to_datetime(df_rd['采样时间'])

# Sort the Excel file in chronological order
df_rd_rank = df_rd.sort_values(by='分诊签到时间', ascending=True)

# Calculate the waiting time, and count by minutes. If it is a radiology examination, replace "分诊签到时间" with "签到/登记时间"，replace "采样时间" with "上机检查时间"
df_rd_rank['waiting time'] = (df_rd_rank['采样时间'] - df_rd_rank['分诊签到时间']).dt.total_seconds() / 60

# Exclude the rows where the "waiting time" is negative
df_rd_rank = df_rd_rank[df_rd['waiting time'] >= 0]

# Extract the month, day and time information and add it to a new column
df_rd_rank['month'] = df_rd_rank['分诊签到时间'].dt.month
df_rd_rank['day'] = df_rd_rank['分诊签到时间'].dt.day
df_rd_rank['hour'] = df_rd_rank['分诊签到时间'].dt.hour
df_rd_rank['week'] = df_rd_rank['分诊签到时间'].dt.weekday

# Count the arrival rate, based on "month", "day" and "hour"
grouped = df_rd_rank.groupby(['month', 'day', 'hour']).size().reset_index(name='arrival rate')

# Calculate the number of queuing patient，If it is a radiology examination, replace "分诊签到时间" with "签到/登记时间"，replace "采样时间" with "上机检查时间"
for i in range(len(df_rd_rank)):
    current_sign_in_time = df_rd_rank.loc[i, '分诊签到时间']
    start_index = max(0, i - 10)   # It needs to be dynamically adjusted, ranging from 10 to 100, depending on the congestion level of the queue scene
    count = (df_rd_rank.loc[start_index:i-1, '采样时间'] > current_sign_in_time).sum()
    df_rd_rank.loc[i, 'the number of queuing patient'] = count

# Remove noise record
q25 = df_rd_rank['waiting time'].quantile(0.25)
q75 = df_rd_rank['waiting time'].quantile(0.75)
iqr = q75 - q25
noise_threshold = q75 + 1.5 * iqr
df_rd_rank_denosie = df_rd_rank[df_rd_rank['waiting time'] <= noise_threshold]

# Save the processed data as a new Excel table
df_rd_rank_denosie.to_excel('1F咽拭子-去重-计算A+C.xlsx', index=False)
