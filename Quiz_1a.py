import pandas as pd
import statsmodels.api as sm
import numpy as np

def get_fff_returns():
    """
    Fetches Fama-French 3-Factor Monthly Data directly from Ken French's website.
    This replaces the custom `erk.get_fff_returns()` function.
    """
    ff_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"
    
    # Read the data, skipping the initial descriptive lines.
    # The monthly data block starts on line 5 (index 4).
    # We first find the end of the monthly data block to avoid reading annual data.
    try:
        # Step 1: Find the end row index of the monthly data block
        raw_data = pd.read_csv(ff_url, compression='zip', header=None, encoding='utf-8')
        monthly_end_index = raw_data[raw_data.iloc[:, 0].str.contains(r'^\s*Ann.\s*Ret.', na=False, regex=True)].index.min()
    except Exception as e:
        return f"Error downloading Fama-French data: {e}"

    nrows = None
    if pd.notna(monthly_end_index):
        # Data starts at row 5 (index 4), so nrows = end_index - 4
        nrows = monthly_end_index - 4
        
    # Step 2: Read the data properly
    fff = pd.read_csv(
        ff_url,
        compression='zip',
        skiprows=4,       # Skip 4 rows of description/blank space
        header=0,         # The 5th row is the column header
        index_col=0,
        na_values=-99.99, # Known missing value marker
        nrows=nrows       # Read only the monthly data block
    )

    # Step 3: Convert the index (YYYYMM) to a PeriodIndex
    fff.index = pd.to_datetime(fff.index, format='%Y%m').to_period('M')
    fff = fff.rename_axis('Date')
    
    return fff

# --- Original (and Fixed) Script Logic ---

# Load industry returns
# FIX 1: Changed "ind49_m_vw_rets.csv" to "ind30_m_vw_rets.csv" to match the uploaded file.
Ind_49 = pd.read_csv(
    "ind30_m_vw_rets.csv",
    index_col=0,
    parse_dates=True,
    date_format="%Y%m"
)

# Convert index to PeriodIndex to correctly handle monthly data
Ind_49 = Ind_49.loc["1991":"2018"].to_period("M")
# Normalize column names for consistent access
Ind_49.columns = Ind_49.columns.str.lower().str.strip()

# Load Fama-French data
fff = get_fff_returns()

if isinstance(fff, str):
    # This branch handles the error message if the FFF download failed
    raise RuntimeError(fff) 

Period = slice("1991", "2018")

# Data is in percentages, so divide by 100 to get decimals
RF = fff.loc[Period, ["RF"]] / 100
Mkt_Excess = fff.loc[Period, ["Mkt-RF"]] / 100

Beer_Rts = Ind_49.loc[Period, ["beer"]] / 100

# Align the returns and the risk-free rate to ensure they have the same dates
Beer_Rts, RF = Beer_Rts.align(RF, join="inner", axis=0)
Beer_Excess = Beer_Rts - RF

# Ensure Mkt_Excess aligns with the newly filtered Beer_Excess index
Mkt_Excess = Mkt_Excess.loc[Beer_Excess.index]

# Regression
# Drop NaN values in the Beer_Excess series
Y = Beer_Excess["beer"].dropna()
# Align the independent variables (Mkt_Excess) with the clean dependent variable (Y)
X = sm.add_constant(Mkt_Excess.loc[Y.index])

# FIX 2: Completed the OLS call with the independent variable (X) and added .fit().
lm = sm.OLS(Y, X).fit()

# Print the regression summary to see the results
print(lm.summary())
