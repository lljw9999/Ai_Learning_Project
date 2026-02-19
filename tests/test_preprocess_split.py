import pandas as pd

from data.preprocess import _time_based_leave_two_out


def test_time_based_leave_two_out_no_leakage() -> None:
    df = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 1, 2, 2, 2],
            "item_id": [10, 11, 12, 13, 20, 21, 22],
            "rating": [5, 5, 5, 5, 4, 4, 5],
            "timestamp": [1, 2, 3, 4, 10, 11, 12],
        }
    )

    train_df, val_df, test_df = _time_based_leave_two_out(df)

    for user in sorted(df["user_id"].unique()):
        user_train = train_df.loc[train_df["user_id"] == user, "timestamp"]
        user_val = val_df.loc[val_df["user_id"] == user, "timestamp"]
        user_test = test_df.loc[test_df["user_id"] == user, "timestamp"]

        assert len(user_val) == 1
        assert len(user_test) == 1
        assert user_train.max() < user_val.iloc[0]
        assert user_val.iloc[0] < user_test.iloc[0]
