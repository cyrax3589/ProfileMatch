from sklearn.linear_model import LogisticRegression


def train_model(
    users,
    feedback,
    similarity_matrix,
    mbti_score,
    location_score
):

    X = []
    y = []

    for _, row in feedback.iterrows():

        try:

            u1 = users[
                users["user_id"] == row["user_id"]
            ].index[0]

            u2 = users[
                users["user_id"] ==
                row["matched_user_id"]
            ].index[0]

            text_sim = similarity_matrix[u1][u2]

            mbti = mbti_score(
                users.loc[u1, "mbti"],
                users.loc[u2, "mbti"]
            )

            location = location_score(
                users.loc[u1, "location"],
                users.loc[u2, "location"]
            )

            X.append([
                text_sim,
                mbti,
                location
            ])

            y.append(row["action"])

        except:

            continue

    model = LogisticRegression()

    model.fit(X, y)

    return model