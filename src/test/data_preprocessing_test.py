from data_preprocessing import *

def test_map_feature_n_of_columns():
    X = np.array([
        [1, 2],
        [3, 4],
        [5, 6]])
    result = map_feature(X[:, 0], X[:, 1], 6)
    assert result.shape == (3, 28)
    
def test_map_feature_values():
    X = np.array([
        [1, 2],
        [3, 4],
        [5, 6]])
    result = map_feature(X[:, 0], X[:, 1], 2)
    assert np.array_equal(
        result,
        np.array([
            [1, 1, 2, 1, 2, 4],
            [1, 3, 4, 9, 12, 16],
            [1, 5, 6, 25, 30, 36]]))
