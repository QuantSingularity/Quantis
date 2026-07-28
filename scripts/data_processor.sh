#!/bin/bash
# data_processor.sh - Automated data processing script for Quantis project
#
# This script automates data processing tasks for the Quantis project:
# - Data ingestion from various sources
# - Data cleaning and validation
# - Feature engineering
# - Data transformation and export
#
# Usage: ./data_processor.sh [options]
# Options:
#   --ingest              Ingest data from sources
#   --clean               Clean and validate data
#   --transform           Transform data and engineer features
#   --export              Export processed data
#   --all                 Perform all data processing steps
#   --source TYPE         Specify data source type (market, alternative, etc.)
#   --output DIR          Output directory for processed data
#   --help                Show this help message
#
# Author: Abrar Ahmed
# Date: May 22, 2025

set -e  # Exit immediately if a command exits with a non-zero status

# Colors for terminal output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default settings
INGEST_DATA=false
CLEAN_DATA=false
TRANSFORM_DATA=false
EXPORT_DATA=false
DATA_SOURCE="market"
# Resolve the actual repository root instead of trusting the caller's
# current directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="$PROJECT_ROOT/data/processed"

# Function to display help message
show_help() {
    echo -e "${BLUE}Data Processing Script for Quantis Project${NC}"
    echo ""
    echo "Usage: ./data_processor.sh [options]"
    echo ""
    echo "Options:"
    echo "  --ingest              Ingest data from sources"
    echo "  --clean               Clean and validate data"
    echo "  --transform           Transform data and engineer features"
    echo "  --export              Export processed data"
    echo "  --all                 Perform all data processing steps"
    echo "  --source TYPE         Specify data source type (market, alternative, etc.)"
    echo "  --output DIR          Output directory for processed data"
    echo "  --help                Show this help message"
    echo ""
    exit 0
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check required dependencies
check_dependencies() {
    echo -e "${BLUE}Checking data processing dependencies...${NC}"

    # Check Python
    if ! command_exists python3; then
        echo -e "${RED}Error: Python 3 is required but not installed.${NC}"
        exit 1
    fi

    # Check pandas
    if ! python3 -c "import pandas" &>/dev/null; then
        echo -e "${YELLOW}Warning: pandas is not installed. Installing...${NC}"
        pip install pandas
    fi

    # Check numpy
    if ! python3 -c "import numpy" &>/dev/null; then
        echo -e "${YELLOW}Warning: numpy is not installed. Installing...${NC}"
        pip install numpy
    fi

    # Check scikit-learn for feature engineering
    if ! python3 -c "import sklearn" &>/dev/null; then
        echo -e "${YELLOW}Warning: scikit-learn is not installed. Installing...${NC}"
        pip install scikit-learn
    fi

    # Check financial libraries
    if ! python3 -c "import pandas_ta" &>/dev/null; then
        echo -e "${YELLOW}Warning: pandas-ta is not installed. Installing...${NC}"
        pip install pandas-ta
    fi

    echo -e "${GREEN}All required data processing dependencies are installed.${NC}"
}

# Function to prepare output directory
prepare_output_dir() {
    echo -e "${BLUE}Preparing output directory...${NC}"

    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR/raw"
    mkdir -p "$OUTPUT_DIR/cleaned"
    mkdir -p "$OUTPUT_DIR/transformed"
    mkdir -p "$OUTPUT_DIR/features"
    mkdir -p "$OUTPUT_DIR/final"

    echo -e "${GREEN}Output directory prepared: $OUTPUT_DIR${NC}"
}

# Function to ingest data
ingest_data() {
    echo -e "${BLUE}Ingesting data from $DATA_SOURCE source...${NC}"

    # Create Python script for data ingestion
    INGEST_SCRIPT="$PROJECT_ROOT/data_ingest.py"

    cat > "$INGEST_SCRIPT" << EOF
#!/usr/bin/env python3
"""
Data Ingestion Script for Quantis Project
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Configuration
DATA_SOURCE = "$DATA_SOURCE"
OUTPUT_DIR = "$OUTPUT_DIR/raw"
START_DATE = datetime.now() - timedelta(days=365)  # 1 year of data
END_DATE = datetime.now()
SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "JNJ"]

def ingest_market_data():
    """Ingest market data from various sources"""
    print(f"Ingesting market data for {len(SYMBOLS)} symbols...")

    # In a real scenario, this would connect to market data APIs
    # For demonstration, we'll generate synthetic data

    for symbol in SYMBOLS:
        print(f"Processing {symbol}...")

        # Generate date range
        dates = pd.date_range(start=START_DATE, end=END_DATE, freq='D')

        # Generate synthetic price data
        np.random.seed(sum(ord(c) for c in symbol))  # Seed based on symbol name
        base_price = np.random.uniform(50, 500)
        volatility = np.random.uniform(0.01, 0.05)

        # Generate OHLCV data
        n_days = len(dates)
        closes = np.zeros(n_days)
        closes[0] = base_price

        for i in range(1, n_days):
            closes[i] = closes[i-1] * (1 + np.random.normal(0, volatility))

        # Generate open, high, low based on close
        opens = closes * (1 + np.random.normal(0, volatility/2, n_days))
        highs = np.maximum(opens, closes) * (1 + np.abs(np.random.normal(0, volatility, n_days)))
        lows = np.minimum(opens, closes) * (1 - np.abs(np.random.normal(0, volatility, n_days)))
        volumes = np.random.normal(1000000, 500000, n_days)
        volumes = np.maximum(volumes, 100000)  # Ensure positive volume

        # Create DataFrame
        df = pd.DataFrame({
            'date': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes,
            'symbol': symbol
        })

        # Save to CSV
        output_file = os.path.join(OUTPUT_DIR, f"{symbol}_market_data.csv")
        df.to_csv(output_file, index=False)
        print(f"Saved {len(df)} records to {output_file}")

def ingest_alternative_data():
    """Ingest alternative data sources"""
    print("Ingesting alternative data...")

    # In a real scenario, this would connect to alternative data APIs
    # For demonstration, we'll generate synthetic data

    # Generate sentiment data
    dates = pd.date_range(start=START_DATE, end=END_DATE, freq='D')

    for symbol in SYMBOLS:
        print(f"Processing sentiment data for {symbol}...")

        # Generate synthetic sentiment data
        np.random.seed(sum(ord(c) for c in symbol) + 1)  # Different seed

        sentiments = np.random.normal(0, 1, len(dates))
        volumes = np.random.normal(1000, 500, len(dates))
        volumes = np.maximum(volumes, 100)  # Ensure positive volume

        # Create DataFrame
        df = pd.DataFrame({
            'date': dates,
            'sentiment_score': sentiments,
            'mention_volume': volumes,
            'symbol': symbol
        })

        # Save to CSV
        output_file = os.path.join(OUTPUT_DIR, f"{symbol}_sentiment_data.csv")
        df.to_csv(output_file, index=False)
        print(f"Saved {len(df)} records to {output_file}")

def ingest_economic_data():
    """Ingest economic indicators data"""
    print("Ingesting economic data...")

    # Generate economic indicator data
    dates = pd.date_range(start=START_DATE, end=END_DATE, freq='D')

    # List of economic indicators
    indicators = [
        "GDP_Growth", "Unemployment_Rate", "Inflation_Rate",
        "Interest_Rate", "Consumer_Confidence"
    ]

    # Generate synthetic data for each indicator
    for indicator in indicators:
        print(f"Processing {indicator}...")

        # Different seed for each indicator
        np.random.seed(sum(ord(c) for c in indicator))

        # Generate values based on indicator type
        if indicator == "GDP_Growth":
            values = np.random.normal(2.5, 0.5, len(dates))
        elif indicator == "Unemployment_Rate":
            values = np.random.normal(5.0, 0.3, len(dates))
        elif indicator == "Inflation_Rate":
            values = np.random.normal(2.0, 0.2, len(dates))
        elif indicator == "Interest_Rate":
            values = np.random.normal(3.0, 0.1, len(dates))
        elif indicator == "Consumer_Confidence":
            values = np.random.normal(100, 5, len(dates))

        # Create DataFrame
        df = pd.DataFrame({
            'date': dates,
            'value': values,
            'indicator': indicator
        })

        # Save to CSV
        output_file = os.path.join(OUTPUT_DIR, f"{indicator}_data.csv")
        df.to_csv(output_file, index=False)
        print(f"Saved {len(df)} records to {output_file}")

def main():
    """Main function to ingest data based on source type"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if DATA_SOURCE == "market" or DATA_SOURCE == "all":
        ingest_market_data()

    if DATA_SOURCE == "alternative" or DATA_SOURCE == "all":
        ingest_alternative_data()

    if DATA_SOURCE == "economic" or DATA_SOURCE == "all":
        ingest_economic_data()

    print("Data ingestion completed successfully.")

if __name__ == "__main__":
    main()
EOF

    # Make script executable
    chmod +x "$INGEST_SCRIPT"

    # Run the ingestion script
    python3 "$INGEST_SCRIPT"

    # Clean up
    rm "$INGEST_SCRIPT"

    echo -e "${GREEN}Data ingestion completed.${NC}"
}

# Function to clean data
clean_data() {
    echo -e "${BLUE}Cleaning and validating data...${NC}"

    # Create Python script for data cleaning
    CLEAN_SCRIPT="$PROJECT_ROOT/data_clean.py"

    cat > "$CLEAN_SCRIPT" << EOF
#!/usr/bin/env python3
"""
Data Cleaning Script for Quantis Project
"""
import os
import sys
import pandas as pd
import numpy as np
import glob
from datetime import datetime

# Configuration
INPUT_DIR = "$OUTPUT_DIR/raw"
OUTPUT_DIR = "$OUTPUT_DIR/cleaned"

def clean_market_data():
    """Clean market data files"""
    print("Cleaning market data...")

    # Find all market data files
    market_files = glob.glob(os.path.join(INPUT_DIR, "*_market_data.csv"))

    for file in market_files:
        symbol = os.path.basename(file).split("_")[0]
        print(f"Cleaning market data for {symbol}...")

        # Read data
        df = pd.read_csv(file)

        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])

        # Sort by date
        df = df.sort_values('date')

        # Handle missing values
        df = df.dropna()

        # Ensure data types
        df['open'] = pd.to_numeric(df['open'], errors='coerce')
        df['high'] = pd.to_numeric(df['high'], errors='coerce')
        df['low'] = pd.to_numeric(df['low'], errors='coerce')
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')

        # Validate data
        # Ensure high >= open, close, low
        df['high'] = df[['high', 'open', 'close']].max(axis=1)
        # Ensure low <= open, close, high
        df['low'] = df[['low', 'open', 'close']].min(axis=1)
        # Ensure volume is positive
        df['volume'] = df['volume'].abs()

        # Remove duplicates
        df = df.drop_duplicates(subset=['date', 'symbol'])

        # Save cleaned data
        output_file = os.path.join(OUTPUT_DIR, f"{symbol}_market_data_clean.csv")
        df.to_csv(output_file, index=False)
        print(f"Saved {len(df)} cleaned records to {output_file}")

def clean_alternative_data():
    """Clean alternative data files"""
    print("Cleaning alternative data...")

    # Find all sentiment data files
    sentiment_files = glob.glob(os.path.join(INPUT_DIR, "*_sentiment_data.csv"))

    for file in sentiment_files:
        symbol = os.path.basename(file).split("_")[0]
        print(f"Cleaning sentiment data for {symbol}...")

        # Read data
        df = pd.read_csv(file)

        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])

        # Sort by date
        df = df.sort_values('date')

        # Handle missing values
        df = df.dropna()

        # Ensure data types
        df['sentiment_score'] = pd.to_numeric(df['sentiment_score'], errors='coerce')
        df['mention_volume'] = pd.to_numeric(df['mention_volume'], errors='coerce')

        # Validate data
        # Normalize sentiment scores to [-1, 1]
        max_abs = max(abs(df['sentiment_score'].min()), abs(df['sentiment_score'].max()))
        if max_abs > 0:
            df['sentiment_score'] = df['sentiment_score'] / max_abs

        # Ensure mention volume is positive
        df['mention_volume'] = df['mention_volume'].abs()

        # Remove duplicates
        df = df.drop_duplicates(subset=['date', 'symbol'])

        # Save cleaned data
        output_file = os.path.join(OUTPUT_DIR, f"{symbol}_sentiment_data_clean.csv")
        df.to_csv(output_file, index=False)
        print(f"Saved {len(df)} cleaned records to {output_file}")

def clean_economic_data():
    """Clean economic data files"""
    print("Cleaning economic data...")

    # Find all economic data files
    economic_files = glob.glob(os.path.join(INPUT_DIR, "*_data.csv"))
    economic_files = [f for f in economic_files if "_market_data.csv" not in f and "_sentiment_data.csv" not in f]

    for file in economic_files:
        indicator = os.path.basename(file).split("_data")[0]
        print(f"Cleaning economic data for {indicator}...")

        # Read data
        df = pd.read_csv(file)

        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])

        # Sort by date
        df = df.sort_values('date')

        # Handle missing values
        df = df.dropna()

        # Ensure data types
        df['value'] = pd.to_numeric(df['value'], errors='coerce')

        # Remove duplicates
        df = df.drop_duplicates(subset=['date', 'indicator'])

        # Save cleaned data
        output_file = os.path.join(OUTPUT_DIR, f"{indicator}_data_clean.csv")
        df.to_csv(output_file, index=False)
        print(f"Saved {len(df)} cleaned records to {output_file}")

def main():
    """Main function to clean data"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Clean market data
    clean_market_data()

    # Clean alternative data
    clean_alternative_data()

    # Clean economic data
    clean_economic_data()

    print("Data cleaning completed successfully.")

if __name__ == "__main__":
    main()
EOF

    # Make script executable
    chmod +x "$CLEAN_SCRIPT"

    # Run the cleaning script
    python3 "$CLEAN_SCRIPT"

    # Clean up
    rm "$CLEAN_SCRIPT"

    echo -e "${GREEN}Data cleaning completed.${NC}"
}

# Function to transform data and engineer features
transform_data() {
    echo -e "${BLUE}Transforming data and engineering features...${NC}"

    # Create Python script for data transformation
    TRANSFORM_SCRIPT="$PROJECT_ROOT/data_transform.py"

    cat > "$TRANSFORM_SCRIPT" << EOF
#!/usr/bin/env python3
"""
Data Transformation and Feature Engineering Script for Quantis Project
"""
import os
import sys
import pandas as pd
import numpy as np
import glob
from datetime import datetime
import pandas_ta as ta

# Configuration
INPUT_DIR = "$OUTPUT_DIR/cleaned"
OUTPUT_DIR_TRANSFORM = "$OUTPUT_DIR/transformed"
OUTPUT_DIR_FEATURES = "$OUTPUT_DIR/features"

def transform_market_data():
    """Transform market data and engineer features"""
    print("Transforming market data and engineering features...")

    # Find all cleaned market data files
    market_files = glob.glob(os.path.join(INPUT_DIR, "*_market_data_clean.csv"))

    for file in market_files:
        symbol = os.path.basename(file).split("_")[0]
        print(f"Processing {symbol}...")

        # Read data
        df = pd.read_csv(file)

        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])

        # Sort by date
        df = df.sort_values('date')

        # Calculate returns
        df['daily_return'] = df['close'].pct_change()
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))

        # Calculate volatility (20-day rolling standard deviation of returns)
        df['volatility_20d'] = df['daily_return'].rolling(window=20).std()

        # Calculate trading volume metrics
        df['volume_change'] = df['volume'].pct_change()
        df['volume_ma_10d'] = df['volume'].rolling(window=10).mean()
        df['relative_volume'] = df['volume'] / df['volume_ma_10d']

        # Save transformed data
        output_file = os.path.join(OUTPUT_DIR_TRANSFORM, f"{symbol}_market_data_transformed.csv")
        df.to_csv(output_file, index=False)
        print(f"Saved transformed data to {output_file}")

        # Engineer technical indicators using pandas_ta
        # Create a clean DataFrame for technical analysis
        df_ta = df[['date', 'open', 'high', 'low', 'close', 'volume']].copy()

        # Calculate moving averages
        df_ta.ta.sma(length=20, append=True)
        df_ta.ta.sma(length=50, append=True)
        df_ta.ta.sma(length=200, append=True)
        df_ta.ta.ema(length=20, append=True)

        # Calculate RSI
        df_ta.ta.rsi(length=14, append=True)

        # Calculate MACD
        df_ta.ta.macd(fast=12, slow=26, signal=9, append=True)

        # Calculate Bollinger Bands
        df_ta.ta.bbands(length=20, std=2, append=True)

        # Calculate Average True Range (ATR)
        df_ta.ta.atr(length=14, append=True)

        # Calculate On-Balance Volume (OBV)
        df_ta.ta.obv(append=True)

        # Calculate Stochastic Oscillator
        df_ta.ta.stoch(append=True)

        # Save feature-engineered data
        output_file = os.path.join(OUTPUT_DIR_FEATURES, f"{symbol}_technical_features.csv")
        df_ta.to_csv(output_file, index=False)
        print(f"Saved technical features to {output_file}")

def merge_market_and_alternative_data():
    """Merge market and alternative data"""
    print("Merging market and alternative data...")

    # Find all symbols with both market and sentiment data
    market_files = glob.glob(os.path.join(INPUT_DIR, "*_market_data_clean.csv"))
    sentiment_files = glob.glob(os.path.join(INPUT_DIR, "*_sentiment_data_clean.csv"))

    market_symbols = [os.path.basename(f).split("_")[0] for f in market_files]
    sentiment_symbols = [os.path.basename(f).split("_")[0] for f in sentiment_files]

    # Find common symbols
    common_symbols = list(set(market_symbols) & set(sentiment_symbols))

    for symbol in common_symbols:
        print(f"Merging data for {symbol}...")

        # Read market data
        market_file = os.path.join(INPUT_DIR, f"{symbol}_market_data_clean.csv")
        market_df = pd.read_csv(market_file)
        market_df['date'] = pd.to_datetime(market_df['date'])

        # Read sentiment data
        sentiment_file = os.path.join(INPUT_DIR, f"{symbol}_sentiment_data_clean.csv")
        sentiment_df = pd.read_csv(sentiment_file)
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])

        # Merge data on date
        merged_df = pd.merge(market_df, sentiment_df, on=['date', 'symbol'], how='outer')

        # Sort by date
        merged_df = merged_df.sort_values('date')

        # Fill missing values
        merged_df = merged_df.ffill()

        # Save merged data
        output_file = os.path.join(OUTPUT_DIR_TRANSFORM, f"{symbol}_merged_data.csv")
        merged_df.to_csv(output_file, index=False)
        print(f"Saved merged data to {output_file}")

def create_master_dataset():
    """Create a master dataset with all symbols and features"""
    print("Creating master dataset...")

    # Find all transformed market data files
    transformed_files = glob.glob(os.path.join(OUTPUT_DIR_TRANSFORM, "*_market_data_transformed.csv"))

    # Read and combine all files
    dfs = []
    for file in transformed_files:
        df = pd.read_csv(file)
        dfs.append(df)

    # Concatenate all dataframes
    if dfs:
        master_df = pd.concat(dfs, ignore_index=True)

        # Sort by symbol and date
        master_df = master_df.sort_values(['symbol', 'date'])

        # Save master dataset
        output_file = os.path.join(OUTPUT_DIR_TRANSFORM, "master_market_dataset.csv")
        master_df.to_csv(output_file, index=False)
        print(f"Saved master market dataset to {output_file}")
    else:
        print("No transformed market data files found.")

    # Find all technical feature files
    feature_files = glob.glob(os.path.join(OUTPUT_DIR_FEATURES, "*_technical_features.csv"))

    # Read and combine all files
    dfs = []
    for file in feature_files:
        df = pd.read_csv(file)
        symbol = os.path.basename(file).split("_")[0]
        df['symbol'] = symbol
        dfs.append(df)

    # Concatenate all dataframes
    if dfs:
        master_df = pd.concat(dfs, ignore_index=True)

        # Sort by symbol and date
        master_df = master_df.sort_values(['symbol', 'date'])

        # Save master dataset
        output_file = os.path.join(OUTPUT_DIR_FEATURES, "master_technical_features.csv")
        master_df.to_csv(output_file, index=False)
        print(f"Saved master technical features dataset to {output_file}")
    else:
        print("No technical feature files found.")

def main():
    """Main function to transform data and engineer features"""
    os.makedirs(OUTPUT_DIR_TRANSFORM, exist_ok=True)
    os.makedirs(OUTPUT_DIR_FEATURES, exist_ok=True)

    # Transform market data and engineer features
    transform_market_data()

    # Merge market and alternative data
    merge_market_and_alternative_data()

    # Create master dataset
    create_master_dataset()

    print("Data transformation and feature engineering completed successfully.")

if __name__ == "__main__":
    main()
EOF

    # Make script executable
    chmod +x "$TRANSFORM_SCRIPT"

    # Run the transformation script
    python3 "$TRANSFORM_SCRIPT"

    # Clean up
    rm "$TRANSFORM_SCRIPT"

    echo -e "${GREEN}Data transformation and feature engineering completed.${NC}"
}

# Function to export processed data
export_data() {
    echo -e "${BLUE}Exporting processed data...${NC}"

    # Create Python script for data export
    EXPORT_SCRIPT="$PROJECT_ROOT/data_export.py"

    cat > "$EXPORT_SCRIPT" << EOF
#!/usr/bin/env python3
"""
Data Export Script for Quantis Project
"""
import os
import sys
import pandas as pd
import numpy as np
import glob
from datetime import datetime
import json

# Configuration
INPUT_DIR_TRANSFORM = "$OUTPUT_DIR/transformed"
INPUT_DIR_FEATURES = "$OUTPUT_DIR/features"
OUTPUT_DIR = "$OUTPUT_DIR/final"

def export_to_csv():
    """Export data to CSV format"""
    print("Exporting data to CSV format...")

    # Find all master datasets
    master_files = [
        os.path.join(INPUT_DIR_TRANSFORM, "master_market_dataset.csv"),
        os.path.join(INPUT_DIR_FEATURES, "master_technical_features.csv")
    ]

    for file in master_files:
        if os.path.exists(file):
            print(f"Exporting {os.path.basename(file)}...")

            # Read data
            df = pd.read_csv(file)

            # Export to CSV
            output_file = os.path.join(OUTPUT_DIR, os.path.basename(file))
            df.to_csv(output_file, index=False)
            print(f"Exported to {output_file}")

def export_to_json():
    """Export data to JSON format"""
    print("Exporting data to JSON format...")

    # Find all master datasets
    master_files = [
        os.path.join(INPUT_DIR_TRANSFORM, "master_market_dataset.csv"),
        os.path.join(INPUT_DIR_FEATURES, "master_technical_features.csv")
    ]

    for file in master_files:
        if os.path.exists(file):
            print(f"Exporting {os.path.basename(file)} to JSON...")

            # Read data
            df = pd.read_csv(file)

            # Convert date to string for JSON serialization
            if 'date' in df.columns:
                df['date'] = df['date'].astype(str)

            # Export to JSON
            base_name = os.path.splitext(os.path.basename(file))[0]
            output_file = os.path.join(OUTPUT_DIR, f"{base_name}.json")

            # Group by symbol for better organization
            result = {}
            for symbol in df['symbol'].unique():
                symbol_data = df[df['symbol'] == symbol].to_dict(orient='records')
                result[symbol] = symbol_data

            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)

            print(f"Exported to {output_file}")

def export_to_sqlite():
    """Export data to SQLite database"""
    print("Exporting data to SQLite database...")

    import sqlite3

    # Create SQLite database
    db_file = os.path.join(OUTPUT_DIR, "quantis_data.db")
    conn = sqlite3.connect(db_file)

    # Find all master datasets
    master_files = [
        os.path.join(INPUT_DIR_TRANSFORM, "master_market_dataset.csv"),
        os.path.join(INPUT_DIR_FEATURES, "master_technical_features.csv")
    ]

    for file in master_files:
        if os.path.exists(file):
            print(f"Exporting {os.path.basename(file)} to SQLite...")

            # Read data
            df = pd.read_csv(file)

            # Determine table name
            base_name = os.path.splitext(os.path.basename(file))[0]
            table_name = base_name.lower().replace('-', '_')

            # Export to SQLite
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            print(f"Exported to table {table_name} in {db_file}")

    # Close connection
    conn.close()

def create_data_catalog():
    """Create a catalog of available datasets"""
    print("Creating data catalog...")

    catalog = {
        "datasets": [],
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_directory": OUTPUT_DIR
    }

    # Find all files in the output directory
    files = glob.glob(os.path.join(OUTPUT_DIR, "*"))

    for file in files:
        file_info = {
            "filename": os.path.basename(file),
            "path": file,
            "size_bytes": os.path.getsize(file),
            "last_modified": datetime.fromtimestamp(os.path.getmtime(file)).strftime("%Y-%m-%d %H:%M:%S"),
            "format": os.path.splitext(file)[1][1:]  # Get extension without dot
        }

        # Add file-specific information
        if file.endswith(".csv"):
            try:
                df = pd.read_csv(file)
                file_info["rows"] = len(df)
                file_info["columns"] = list(df.columns)
                file_info["symbols"] = list(df["symbol"].unique()) if "symbol" in df.columns else []
                file_info["date_range"] = [
                    df["date"].min(),
                    df["date"].max()
                ] if "date" in df.columns else []
            except Exception as e:
                file_info["error"] = str(e)

        catalog["datasets"].append(file_info)

    # Save catalog to JSON
    catalog_file = os.path.join(OUTPUT_DIR, "data_catalog.json")
    with open(catalog_file, 'w') as f:
        json.dump(catalog, f, indent=2)

    print(f"Data catalog created at {catalog_file}")

def main():
    """Main function to export processed data"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Export to different formats
    export_to_csv()
    export_to_json()

    try:
        export_to_sqlite()
    except ImportError:
        print("SQLite export skipped: sqlite3 module not available.")

    # Create data catalog
    create_data_catalog()

    print("Data export completed successfully.")

if __name__ == "__main__":
    main()
EOF

    # Make script executable
    chmod +x "$EXPORT_SCRIPT"

    # Run the export script
    python3 "$EXPORT_SCRIPT"

    # Clean up
    rm "$EXPORT_SCRIPT"

    echo -e "${GREEN}Data export completed.${NC}"
}

# Parse command line arguments
if [ $# -eq 0 ]; then
    show_help
fi

while [ "$1" != "" ]; do
    case $1 in
        --ingest )  INGEST_DATA=true
                    ;;
        --clean )   CLEAN_DATA=true
                    ;;
        --transform ) TRANSFORM_DATA=true
                    ;;
        --export )  EXPORT_DATA=true
                    ;;
        --all )     INGEST_DATA=true
                    CLEAN_DATA=true
                    TRANSFORM_DATA=true
                    EXPORT_DATA=true
                    ;;
        --source )  shift
                    DATA_SOURCE="$1"
                    ;;
        --output )  shift
                    OUTPUT_DIR="$1"
                    ;;
        --help )    show_help
                    ;;
        * )         echo -e "${RED}Error: Unknown option $1${NC}"
                    show_help
                    ;;
    esac
    shift
done

# Main execution
echo -e "${BLUE}Starting Quantis data processing...${NC}"

# Check dependencies
check_dependencies

# Prepare output directory
prepare_output_dir

# Process data
if $INGEST_DATA; then
    ingest_data
fi

if $CLEAN_DATA; then
    clean_data
fi

if $TRANSFORM_DATA; then
    transform_data
fi

if $EXPORT_DATA; then
    export_data
fi

echo -e "${GREEN}Quantis data processing completed successfully!${NC}"
exit 0
