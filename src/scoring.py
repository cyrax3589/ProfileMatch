def mbti_score(mbti1, mbti2):

    if mbti1 == mbti2:
        return 1.0

    elif mbti1[0] == mbti2[0]:
        return 0.7

    else:
        return 0.4


def location_score(loc1, loc2):

    if loc1 == loc2:
        return 1.0

    else:
        return 0.5


def compatibility_score(
    users,
    similarity_matrix,
    i,
    j,
    w1=0.5,
    w2=0.3,
    w3=0.2
):

    text_sim = similarity_matrix[i][j]

    mbti = mbti_score(
        users.loc[i, "mbti"],
        users.loc[j, "mbti"]
    )

    location = location_score(
        users.loc[i, "location"],
        users.loc[j, "location"]
    )

    score = (
        w1 * text_sim +
        w2 * mbti +
        w3 * location
    )

    return round(score * 100, 2)


def get_top_matches(
    users,
    similarity_matrix,
    user_index,
    top_n=5
):

    scores = []

    for j in range(len(users)):

        if j != user_index:

            score = compatibility_score(
                users,
                similarity_matrix,
                user_index,
                j
            )

            scores.append((j, score))

    scores.sort(
        key=lambda x: x[1],
        reverse=True
    )

    top_matches = scores[:top_n]

    results = []

    for i, s in top_matches:

        results.append({
            "name": users.loc[i, "name"],
            "score": s,
            "location": users.loc[i, "location"],
            "profession": users.loc[i, "profession"]
        })

    return results