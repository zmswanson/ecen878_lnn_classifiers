import pandas as pd

class AdultPreprocessing:
    def __init__(self, df):
        self.df = df.copy()
        self.fix_question_marks()

    def __call__(self):
        self.fix_question_marks()
        self.impute_missing_values()
        self.one_hot_encode()
        return self.df.copy()

    def fix_question_marks(self):
        self.df = self.df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        self.df = self.df.replace('?', None)
        
    def impute_missing_values(self, verbose=False):
        '''Imputes missing numerical values with the median and missing categorical values with the mode'''
        # Get the columns with missing values and the number of missing values in each column
        missing_values = self.df.isnull().sum()
        missing_values = missing_values[missing_values > 0]

        # Impute missing values
        for column in missing_values.index:
            if verbose:
                print(f"Column `{column}` has {missing_values[column]} missing values ", end="")

            if self.df[column].dtype == "object":
                self.df[column] = self.df[column].fillna(self.df[column].mode()[0])
                if verbose:
                    print("imputed with mode")
            else:
                self.df[column] = self.df[column].fillna(self.df[column].median())
                if verbose:
                    print("imputed with median")

    def one_hot_encode(self):
        categorical_df = self.df.select_dtypes(include='object')
        onehot_df = pd.get_dummies(categorical_df, drop_first=True).astype(int)
        self.df = self.df.drop(columns=categorical_df.columns)
        self.df = pd.concat([self.df, onehot_df], axis=1)
    
    def get_df(self):
        return self.df.copy()