import numpy as np
import pandas as pd

from random import random, choice
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

POSITIONS = {
    "GK": (["GK"], 1),
    "DF": (["DF", "DFMF", "DFFW"], 4),
    "MF": (["MF", "MFDF", "MFFW"], 3),
    "FW": (["FW", "FWDF", "FWMF"], 3),
}

TO_KEEP = [
    "Player",
    "Goals",
    "Assists",
    "current_value",
]


def create_indiv(df: pd.DataFrame,
                 to_keep: list = TO_KEEP,
                 positions: dict = POSITIONS):
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        to_keep (list, optional): _description_. Defaults to TO_KEEP.
        positions (dict, optional): _description_. Defaults to POSITIONS.

    Returns:
        _type_: _description_
    """
    indiv = pd.DataFrame()
    for pos, (all_pos, n) in positions.items():
        temp = shuffle(df.loc[df["Pos"].isin(all_pos)]).iloc[:n][to_keep]
        temp["Pos"] = pos
        indiv = pd.concat([indiv, temp])
        indiv["Pos"] = pd.Categorical(indiv["Pos"], list(positions.keys()))
        indiv.sort_values(by=["Pos"], inplace=True)

    return indiv


def cross_over(parent_1: dict, parent_2: dict, p: float = 0.5):
    """_summary_

    Args:
        parent_1 (dict): _description_
        parent_2 (dict): _description_
        p (float, optional): _description_. Defaults to 0.5.

    Returns:
        _type_: _description_
    """
    child_1, child_2 = pd.DataFrame(), pd.DataFrame()

    # if we draw a number lower than p...
    if random() < p:
        # ... we recombine the genes
        for pos in parent_1["Pos"].unique():
            concat = pd.concat([
                parent_1.loc[parent_1["Pos"] == pos],
                parent_2.loc[parent_2["Pos"] == pos]
            ])
            it_1, it_2 = train_test_split(concat, test_size=0.5)
            child_1 = pd.concat([child_1, it_1])
            child_2 = pd.concat([child_2, it_2])
    else:
        child_1 = pd.concat([child_1, parent_1])
        child_2 = pd.concat([child_2, parent_2])

    return child_1.sort_values(by=["Pos"]), child_2.sort_values(by=["Pos"])


def mutate(indiv: dict,
           df: pd.DataFrame,
           p_mut: float = 0.5,
           positions: dict = POSITIONS,
           to_keep: list = TO_KEEP):
    """_summary_

    Args:
        indiv (dict): _description_
        df (pd.DataFrame): _description_
        p_mut (float, optional): _description_. Defaults to 0.5.
        positions (dict, optional): _description_. Defaults to POSITIONS.
        to_keep (list, optional): _description_. Defaults to TO_KEEP.

    Returns:
        _type_: _description_
    """
    new_indiv = indiv.copy()

    # if we draw a number lower than p_mut...
    if random() < p_mut:
        # ...we randomly alter a gene
        pos_to_mutate = choice(list(positions.keys()))
        new_gene = df[df["Pos"].isin(
            positions[pos_to_mutate][0])].sample(n=1)[to_keep + ["Pos"]]
        positions[pos_to_mutate][0]
        new_gene["Pos"] = pos_to_mutate
        idx_to_drop = choice(
            new_indiv[new_indiv["Pos"] == pos_to_mutate].index)
        new_indiv.drop(idx_to_drop, inplace=True)
        new_indiv = pd.concat([new_indiv, new_gene])
        new_indiv["Pos"] = pd.Categorical(new_indiv["Pos"],
                                          list(positions.keys()))
        new_indiv.sort_values(by=["Pos"], inplace=True)
    return new_indiv


def objective(
    indiv: pd.DataFrame,
    alpha: float = 0.5,
    beta: float = 0.5,
):
    return alpha * indiv["Goals"].mean() - beta * indiv["current_value"].mean()


def evolution(
    df: pd.DataFrame,
    n_pop: int = 100,
    n_gen: int = 10,
):
    pop = [create_indiv(df) for _ in range(n_pop)]
    for n in range(n_gen):
        scores = [objective(indiv) for indiv in pop]
        print("{:,.0f}th generation | average score : {:,.4f}".format(
            n + 1, np.mean(scores)))
        sorted_pop = [pop[pos] for pos in np.argsort([-s for s in scores])]
        middle = len(sorted_pop) // 2
        parents = sorted_pop[:middle]
        parents_1, parents_2 = train_test_split(parents, test_size=0.5)
        children = [cross_over(x, y) for x, y in zip(parents_1, parents_2)]
        children = [x for y in children for x in y]
        pop = parents + children

    # at the end select the best individual(s)
    scores = [objective(indiv) for indiv in pop]
    return [pop[pos] for pos in np.argsort(scores)][0]
