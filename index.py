import pandas as pd
import chardet
import matplotlib.pyplot as plt
import numpy as np

# Detect file encoding first
def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read(10000))
    return result['encoding']

try:
    # Try to detect encoding automatically
    encoding = detect_encoding("data.csv")
    print(f"Detected encoding: {encoding}")
    
    # Load dataset with detected encoding
    df = pd.read_csv("data.csv", encoding=encoding)
    
    # Clean problematic characters in all string columns
    for col in df.select_dtypes(include=['object']):
        df[col] = df[col].str.replace(r'[^\x00-\x7F]+', '', regex=True)
    
    # Display basic information
    print("\n=== Dataset Information ===")
    print(df.info())
    
    print("\n=== First 5 Rows ===")
    print(df.head())
    
    print("\n=== Missing Values ===")
    print(df.isnull().sum())
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=['number']).columns
    if not numeric_cols.empty:
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    print("\n=== Summary Statistics ===")
    print(df.describe())
    
    # --------------------------
    # Custom Visualizations
    # --------------------------
    plt.style.use('ggplot')
    plt.rcParams['font.size'] = 10
    
    # 1. Distribution of values (log scale)
    plt.figure(figsize=(10, 6))
    plt.hist(df['value'].dropna(), bins=50, color='skyblue', 
             edgecolor='black', log=True)
    plt.title('Distribution of Values (Log Scale)', pad=20)
    plt.xlabel('Value')
    plt.ylabel('Frequency (log)')
    plt.tight_layout()
    plt.show()
    
    # 2. Top 10 industries by average value
    top_industries = df.groupby('industry')['value'].mean().nlargest(10)
    plt.figure(figsize=(12, 6))
    top_industries.sort_values().plot(kind='barh', color='teal')
    plt.title('Top 10 Industries by Average Value', pad=20)
    plt.xlabel('Average Value')
    plt.ylabel('Industry')
    plt.tight_layout()
    plt.show()
    
    # 3. Value distribution by level
    plt.figure(figsize=(10, 6))
    df.boxplot(column='value', by='level', grid=False, 
               showfliers=False,
               patch_artist=True,
               boxprops=dict(facecolor='lightgreen'))
    plt.title('Value Distribution by Level', pad=20)
    plt.suptitle('')
    plt.xlabel('Level')
    plt.ylabel('Value')
    plt.yscale('log')
    plt.tight_layout()
    plt.show()
    
    # 4. IMPROVED Size category analysis (Pie Chart)
    if 'size' in df.columns:
        plt.figure(figsize=(10, 8))
        size_counts = df['size'].value_counts()
        
        # Clean labels
        size_counts.index = size_counts.index.str.replace(r'[^\w\s-]', '', regex=True).str.strip()
        
        # Explode small slices for better visibility
        explode = [0.1 if (count/size_counts.sum())*100 < 5 else 0 
                  for count in size_counts]
        
        # Custom autopct function
        def autopct_format(pct):
            return f'{pct:.1f}%' if pct > 3 else ''
        
        # Create pie
        wedges, texts, autotexts = plt.pie(
            size_counts,
            labels=None,  # We'll use legend instead
            autopct=autopct_format,
            startangle=90,
            pctdistance=0.8,
            explode=explode,
            colors=plt.cm.tab20.colors,
            textprops={'fontsize': 9, 'color': 'black'},
            wedgeprops={'edgecolor': 'white', 'linewidth': 0.5}
        )
        
        plt.title('Business Size Distribution', pad=25, fontsize=12)
        
        # Create comprehensive legend
        legend_labels = [f"{label} ({count:,})" 
                        for label, count in zip(size_counts.index, size_counts)]
        
        plt.legend(
            wedges,
            legend_labels,
            title="Size Categories",
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            fontsize=9
        )
        
        # Equal aspect ratio ensures pie is drawn as circle
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
    
    # 5. Scatter plot of level vs value (log scale)
    plt.figure(figsize=(10, 6))
    plt.scatter(df['level'], df['value'], alpha=0.5, 
               color='purple', s=30)
    plt.yscale('log')
    plt.title('Level vs Value (Log Scale)', pad=20)
    plt.xlabel('Level')
    plt.ylabel('Value (log)')
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"An error occurred: {str(e)}")
