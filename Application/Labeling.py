import pandas as pd
def label_data(row):
    if row['Face'] == 1 and row['Age'] in ['(0-2)', '(4-6)', '(8-13)', '(15-20)', '(25-32)']:
        return 1
    
    elif row['Cartoon'] == 1:
        return 1
    
    elif row['Brightness_Level'] > 5:
        return 1
    
    elif row['Dominant_COLOR'] in ['Red', 'Orange', 'Yellow']:
        return 1
    return 0

df = pd.read_csv('static/marketing_images_processed.csv')
df['Label'] = df.apply(label_data, axis=1)

print(df.head())