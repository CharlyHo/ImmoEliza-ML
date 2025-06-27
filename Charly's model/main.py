import matplotlib
from src.data_cleanner import DataCleanner
from src.data_analysis_plots import data_analysis_charts
from src.surface import generate_surface_charts
from src.most_expensive_region import generate_all_expensive_municipality_charts
from src.less_expensive_region import generate_all_least_expensive_charts

matplotlib.use('TkAgg')

# Initialization and data cleaning
cleaner = DataCleanner("data/immoweb-dataset.csv")
cleaner.send_output_file("data/data_cleanned.csv")

# Convert -1 values to NaN so they are not included in the correlation
df = cleaner.to_real_values() 

# Data analysis
data_analysis_charts(df, True)

# Data interpretation
generate_surface_charts(df, True)

generate_all_expensive_municipality_charts()

generate_all_least_expensive_charts()