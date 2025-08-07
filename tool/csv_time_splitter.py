#!/usr/bin/env python3
"""
CSV Time Splitter Tool

This script processes CSV files in the current directory and splits them into 8 time periods
based on the 'time' column (timestamp in milliseconds). Each time period is saved to a 
separate folder with appropriate naming.

Author: AI Assistant
Date: 2024
"""

import os
import pandas as pd
import logging
from pathlib import Path
from typing import List, Tuple, Dict
import sys
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('csv_processing.log')
    ]
)
logger = logging.getLogger(__name__)

# Define time periods (in milliseconds)
TIME_PERIODS = {
    "测试1": (0, 158000),           # 00:00~02:38
    "测试2": (159000, 273000),      # 02:39~04:33
    "测试3": (274000, 318000),      # 04:34~05:18
    "测试4": (319000, 406000),      # 05:19~06:46
    "测试5": (407000, 483000),      # 06:47~08:03
    "测试6": (486000, 553000),      # 08:06~09:13
    "测试7": (554000, 592000),      # 09:14~09:52
    "测试8": (592000, 646000),      # 09:52~10:46
}


def find_csv_files(directory: str = ".") -> List[str]:
    """
    Find all CSV files in the specified directory.
    
    Args:
        directory: Directory path to search for CSV files
        
    Returns:
        List of CSV file paths
    """
    csv_files = []
    try:
        for file in os.listdir(directory):
            if file.lower().endswith('.csv'):
                csv_files.append(os.path.join(directory, file))
        logger.info(f"Found {len(csv_files)} CSV files in directory: {directory}")
        return csv_files
    except Exception as e:
        logger.error(f"Error finding CSV files in {directory}: {e}")
        return []


def validate_csv_file(file_path: str) -> Tuple[bool, pd.DataFrame]:
    """
    Validate CSV file and check if it has the required 'time' column.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Tuple of (is_valid, dataframe)
    """
    try:
        # Try different encodings
        encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                logger.info(f"Successfully read {file_path} with encoding: {encoding}")
                break
            except UnicodeDecodeError:
                continue
                
        if df is None:
            logger.error(f"Could not read {file_path} with any encoding")
            return False, pd.DataFrame()
            
        # Check if 'time' column exists
        if 'time' not in df.columns:
            logger.error(f"CSV file {file_path} does not contain 'time' column")
            logger.info(f"Available columns: {list(df.columns)}")
            return False, df
            
        # Check if time column has valid data
        if df['time'].isnull().all():
            logger.error(f"CSV file {file_path} has empty 'time' column")
            return False, df
            
        logger.info(f"CSV file {file_path} is valid with {len(df)} rows")
        return True, df
        
    except Exception as e:
        logger.error(f"Error reading CSV file {file_path}: {e}")
        return False, pd.DataFrame()


def create_output_directories() -> Dict[str, str]:
    """
    Create output directories for each time period.
    
    Returns:
        Dictionary mapping period names to directory paths
    """
    directories = {}
    try:
        for period_name in TIME_PERIODS.keys():
            dir_path = os.path.join(".", period_name)
            os.makedirs(dir_path, exist_ok=True)
            directories[period_name] = dir_path
            logger.info(f"Created/verified directory: {dir_path}")
        return directories
    except Exception as e:
        logger.error(f"Error creating directories: {e}")
        return {}


def filter_data_by_time_period(df: pd.DataFrame, period_name: str) -> pd.DataFrame:
    """
    Filter dataframe by time period.
    
    Args:
        df: Input dataframe
        period_name: Name of the time period
        
    Returns:
        Filtered dataframe
    """
    try:
        start_time, end_time = TIME_PERIODS[period_name]
        
        # Filter data within the time range
        filtered_df = df[(df['time'] >= start_time) & (df['time'] <= end_time)]
        
        logger.info(f"Period {period_name} ({start_time}-{end_time}ms): {len(filtered_df)} rows")
        return filtered_df
        
    except Exception as e:
        logger.error(f"Error filtering data for period {period_name}: {e}")
        return pd.DataFrame()


def process_csv_file(file_path: str, output_dirs: Dict[str, str]) -> bool:
    """
    Process a single CSV file and split it into time periods.
    
    Args:
        file_path: Path to the CSV file
        output_dirs: Dictionary of output directories
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Validate and read CSV file
        is_valid, df = validate_csv_file(file_path)
        if not is_valid:
            return False
            
        # Get base filename without extension
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        
        # Process each time period
        total_rows_processed = 0
        
        for period_name, output_dir in output_dirs.items():
            # Filter data for this time period
            filtered_df = filter_data_by_time_period(df, period_name)
            
            if len(filtered_df) > 0:
                # Create output filename with period suffix
                output_filename = f"{base_filename}_{period_name}.csv"
                output_path = os.path.join(output_dir, output_filename)
                
                # Save filtered data
                filtered_df.to_csv(output_path, index=False, encoding='utf-8')
                logger.info(f"Saved {len(filtered_df)} rows to {output_path}")
                total_rows_processed += len(filtered_df)
            else:
                logger.info(f"No data found for period {period_name} in file {file_path}")
        
        logger.info(f"Processed file {file_path}: {total_rows_processed} total rows processed")
        return True
        
    except Exception as e:
        logger.error(f"Error processing CSV file {file_path}: {e}")
        return False


def print_time_periods_info():
    """Print information about the time periods."""
    logger.info("Time period definitions:")
    for period_name, (start, end) in TIME_PERIODS.items():
        start_min = start // 60000
        start_sec = (start % 60000) // 1000
        end_min = end // 60000
        end_sec = (end % 60000) // 1000
        logger.info(f"  {period_name}: {start:,}-{end:,}ms ({start_min:02d}:{start_sec:02d}~{end_min:02d}:{end_sec:02d})")


def main():
    """Main function to process all CSV files."""
    parser = argparse.ArgumentParser(
        description='CSV Time Splitter Tool - Split CSV files into time periods based on timestamp',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Time Period Definitions:
  测试1: 0-158,000ms (00:00~02:38)
  测试2: 159,000-273,000ms (02:39~04:33)
  测试3: 274,000-318,000ms (04:34~05:18)
  测试4: 319,000-406,000ms (05:19~06:46)
  测试5: 407,000-483,000ms (06:47~08:03)
  测试6: 486,000-553,000ms (08:06~09:13)
  测试7: 554,000-592,000ms (09:14~09:52)
  测试8: 592,000-646,000ms (09:52~10:46)

Examples:
  python csv_time_splitter.py
  python csv_time_splitter.py --directory /path/to/csv/files
        """
    )
    
    parser.add_argument(
        '--directory', '-d',
        type=str,
        default='.',
        help='Directory containing CSV files to process (default: current directory)'
    )
    
    parser.add_argument(
        '--version', '-v',
        action='version',
        version='CSV Time Splitter Tool v1.0'
    )
    
    args = parser.parse_args()
    
    logger.info("Starting CSV Time Splitter Tool")
    print_time_periods_info()
    
    # Find CSV files in specified directory
    csv_files = find_csv_files(args.directory)
    
    if not csv_files:
        logger.warning(f"No CSV files found in directory: {args.directory}")
        return
    
    # Create output directories
    output_dirs = create_output_directories()
    if not output_dirs:
        logger.error("Failed to create output directories")
        return
    
    # Process each CSV file
    successful_files = 0
    failed_files = 0
    
    for csv_file in csv_files:
        logger.info(f"Processing file: {csv_file}")
        if process_csv_file(csv_file, output_dirs):
            successful_files += 1
        else:
            failed_files += 1
    
    # Summary
    logger.info("="*50)
    logger.info("Processing Summary:")
    logger.info(f"  Total CSV files found: {len(csv_files)}")
    logger.info(f"  Successfully processed: {successful_files}")
    logger.info(f"  Failed to process: {failed_files}")
    logger.info(f"  Output directories created: {len(output_dirs)}")
    
    if successful_files > 0:
        logger.info("Processing completed successfully!")
    else:
        logger.warning("No files were processed successfully")


if __name__ == "__main__":
    main()