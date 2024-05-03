import datetime
import dill
import pandas as pd

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder


def add_target_column(df_hits, df_session):
    """
    Добавляет целевую переменную

    :param df_hits: датафрайм с целевым действием
    :param df_session: датафрайм для обучения
    :return: датафрайм для обучения с целевой переменной
    """
    target_list = ['sub_car_claim_click', 'sub_car_claim_submit_click',
                   'sub_open_dialog_click', 'sub_custom_question_submit_click',
                   'sub_call_number_click', 'sub_callback_submit_click',
                   'sub_submit_success', 'sub_car_request_submit_click']
    df_hits_prep = df_hits.copy()
    df_hits_prep['target_param'] = df_hits_prep.apply(lambda x: 1 if x.event_action in target_list else 0, axis=1)
    group_id_hits = df_hits_prep.groupby('session_id').agg(target_column=('target_param', 'max'))
    df_count_session = pd.merge(left=df_session, right=group_id_hits, on='session_id', how='inner')
    return df_count_session


def drop_column(df):
    """
    Удаляет столбцы в которых кол-во пропусков больше 20%

    :param df: целевой датафрайм
    :return: исправленный датафрайм
    """
    drop_list = [
        'device_model',
        'utm_keyword',
        'device_os',
        'session_id',
        'client_id',
        'visit_date',
        'visit_time',
        'device_screen_resolution'
    ]
    return df.drop(drop_list, axis=1)


def df_prep(df):
    """
    Замена пропусков

    :param df: целевой датафрайм
    :return: исправленный датафрайм
    """
    device_pc = [
        'Windows',
        'Linux',
        'Chrome OS'
    ]
    df['device_brand'] = df.apply(
        lambda x: 'PC' if x.device_os in device_pc else x.device_brand, axis=1
    )
    df['device_brand'] = df.apply(
        lambda x: 'Apple' if x.device_os == 'Macintosh' else x.device_brand, axis=1
    )
    df['device_brand'] = df.device_brand.fillna('Apple')
    df.utm_campaign = df.utm_campaign.fillna(df.utm_campaign.describe().loc['top'])
    df.utm_adcontent = df.utm_adcontent.fillna(df.utm_adcontent.describe().loc['top'])
    df.utm_source = df.utm_source.fillna(df.utm_source.describe().loc['top'])
    df.utm_medium = df.utm_medium.replace(['(none)', '(not set)'], 'other')
    df.geo_city = df.geo_city.replace('(not set)', 'other')
    df.geo_country = df.geo_country.replace('(not set)', 'other')
    return df


def drop_unique_param(df):
    """
    Функция, которая убирает редкие значения и заменяет их other

    :param df: целевой датафрайм
    :return: исправленный датафрайм
    """
    column_list = [
        'utm_source', 'utm_medium', 'utm_campaign', 'utm_adcontent',
        'device_category', 'device_brand', 'device_browser',
        'geo_country', 'geo_city'
    ]
    for column in column_list:
        series_unigue = df[column].value_counts()
        list_unique = series_unigue[series_unigue <= round(df.shape[0] * 0.0003)].index.tolist()
        df[column] = df.apply(lambda x: 'other' if x[column] in list_unique else x[column], axis=1)
    return df


def df_encoder(df):
    """
    Функция дамми кодирования категориальных данных

    :param df: целевой датафрайм
    :return: исправленный датафрайм
    """
    df_prep = pd.get_dummies(df, columns=[
        'utm_source', 'utm_medium', 'utm_campaign', 'utm_adcontent',
        'device_category', 'device_brand', 'device_browser',
        'geo_country', 'geo_city'
    ], dtype=int)
    return df_prep


def main():
    df_hits = pd.read_csv('data/ga_hits.csv')
    df_session = pd.read_csv('data/ga_sessions.csv', low_memory=False)
    df = add_target_column(df_hits, df_session)

    categorical_features = make_column_selector(dtype_include=object)

    X = df.drop(['target_column'], axis=1)
    y = df.target_column

    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    column_transformer = ColumnTransformer(transformers=[
        ('categorical', categorical_transformer, categorical_features)
    ], verbose_feature_names_out=False)

    preprocessor = Pipeline(steps=[
        ('filter', FunctionTransformer(df_prep)),
        ('outlier_remover', FunctionTransformer(drop_column)),
        ('unique_remove', FunctionTransformer(drop_unique_param)),
        ('encoder', column_transformer)
        # ('encoder', FunctionTransformer(df_encoder))
    ])

    rf_clf = RandomForestClassifier(random_state=42, max_depth=50, max_leaf_nodes=500, class_weight='balanced')

    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', rf_clf)
    ]).fit(X, y)

    score = cross_val_score(rf_pipeline, X, y, cv=4, scoring='roc_auc')
    with open('loan_pipe.pickle', 'wb') as file:
        dill.dump(
            {
                'model': rf_pipeline,
                'metadata': {
                    'name': 'Loan predicton medel',
                    'auther': 'Roman Yadonist',
                    'date': datetime.datetime.now(),
                    'ROC-auc': round(score.mean(), 4)
                }
            },
            file,
            recurse=True
        )


if __name__ == '__main__':
    main()
