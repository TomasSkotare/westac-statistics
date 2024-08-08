import pandas as pd
import numpy as np
from loguru import logger

class DataFrameOptimizer:
    @staticmethod
    def is_feather_compatible(df: pd.DataFrame) -> bool:
        compatible_types = {
            'int64', 'int32', 'int16', 'int8',
            'uint8', 'uint16', 'uint32', 'uint64',
            'Int8', 'Int16', 'Int32', 'Int64',
            'UInt8', 'UInt16', 'UInt32', 'UInt64',
            'float64', 'float32',
            'bool',
            'datetime64[ns]',
            'object',
            'category'
        }    
        return set(df.dtypes.apply(lambda x: x.name).unique()).issubset(compatible_types)

    @staticmethod
    def check_memory_savings(df, column, verbose=False):
        original_memory = df[column].memory_usage(deep=True)

        # Convert to category
        category_series = df[column].astype('category')

        category_memory = category_series.memory_usage(deep=True)

        memory_savings = original_memory - category_memory

        # Check if memory savings are positive
        if memory_savings > 0:
            if verbose:
                logger.debug(f"Converting {column} to 'category' would save {memory_savings} bytes of memory.")
            return True
        else:
            if verbose:
                logger.debug(f"Converting {column} to 'category' would not save memory.")
            return False

    @staticmethod
    def drop_index_like_columns(df: pd.DataFrame, type: str|None=None) -> None:
        if not type:
            return
        # Check each column
        duplicate_columns = DataFrameOptimizer.check_index_duplicates(df)
        for col in duplicate_columns:
            if type is True or type == col:
                df.drop(col, axis=1, inplace=True)
        
        

    @staticmethod
    def downcast_dataframe(df: pd.DataFrame, verbose: bool = False, drop_index_like: str|bool = False) -> pd.DataFrame:
        df_copy = df.copy()

        if verbose:
            logger.debug("Before downcasting:")
            df_copy.info(memory_usage='deep')

        
        DataFrameOptimizer.drop_index_like_columns(df_copy, drop_index_like)

        # Downcast numerical columns
        for col in df_copy.select_dtypes(include=['int']):
            # Check if column has only non-negative values
            if df_copy[col].min() >= 0:
                # Downcast to the smallest possible unsigned integer type
                df_copy[col] = pd.to_numeric(df_copy[col], downcast='unsigned')
            else:
                # Downcast to the smallest possible integer type
                df_copy[col] = pd.to_numeric(df_copy[col], downcast='integer')

        # Downcast float columns separately
        for col in df_copy.select_dtypes(include=['float']):
            # Check if column contains only integers
            if df_copy[col].apply(float.is_integer).all():
                # Downcast to the smallest possible integer type
                df_copy[col] = pd.to_numeric(df_copy[col], downcast='integer')
            else:
                # Downcast to the smallest possible float type
                df_copy[col] = pd.to_numeric(df_copy[col], downcast='float')

        # Change object columns to category if number of unique values is less than half the total number of values
        for col in df_copy.select_dtypes(include=['object']):
            try:
                num_unique_values = len(df_copy[col].unique())
            except TypeError as e:
                # We skip values that cannot be converted to category
                continue
            num_total_values = len(df_copy[col])
            if num_unique_values / num_total_values < 0.7:
                if DataFrameOptimizer.check_memory_savings(df_copy, col, verbose):
                    df_copy[col] = df_copy[col].astype('category')

        # Change boolean columns to bool data type
        for col in df_copy.select_dtypes(include=['bool']):
            df_copy[col] = df_copy[col].astype('bool')

        if verbose:
            logger.debug("After downcasting:")
            df_copy.info(memory_usage='deep')
            if DataFrameOptimizer.is_feather_compatible(df_copy):
                logger.debug("The DataFrame can be saved with Feather.")
            else:
                logger.warning("The DataFrame cannot be saved with Feather.")
            
        return df_copy

    @staticmethod
    def trim_low_weights(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
        if not threshold:
            return df
        # first check if it has a weight column
        if 'weight' not in df.columns:
            logger.warning('No weight column found in dataframe!')
            return df

        initial_count = len(df)
        df = df[df['weight'] > threshold]
        final_count = len(df)

        removed_count = initial_count - final_count
        logger.debug(f'Removed {removed_count} rows ({(removed_count / initial_count) * 100:.2f}%)')

        return df
    
    @staticmethod
    def check_index_duplicates(df: pd.DataFrame) -> list[str]:
        duplicates = [col for col in df.columns if np.all(df[col].values == df.index)]
        return duplicates

    @staticmethod
    def optimize_dataframe(df: pd.DataFrame, threshold: float = None, verbose: bool = False, drop_index_like: bool = False) -> pd.DataFrame:
        df = DataFrameOptimizer.downcast_dataframe(df, verbose, drop_index_like)
        if threshold is not None:
            df = DataFrameOptimizer.trim_low_weights(df, threshold)
        return df
    
    @staticmethod
    def show_df_dict_memory_usage(dfs: dict[str, pd.DataFrame]) -> None:
        for name, df in dfs.items():
            mem_usage_bytes = df.memory_usage(deep=True).sum()
            mem_usage_megabytes = mem_usage_bytes / 1024 ** 2
            print(f"\nMemory usage for {name}: {mem_usage_megabytes:.2f} MB")
            print('Size per columns (MB)')
            print(df.memory_usage(deep=True).apply(lambda x: x / 1024 ** 2))
            print('\n')