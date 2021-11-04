import numpy as np
import matplotlib.pyplot as plt

def display_data(X, example_width=None):
    if not example_width:
        example_width = int(np.round(np.sqrt(X.shape[1])))

    m, n = X.shape
    example_height = int(n / example_width)

    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m / display_rows))
 
    pad = 0
 
    display_array = - np.ones([
        int(pad + display_rows * (example_height + pad)),
        int(pad + display_cols * (example_width + pad))])

    curr_ex = 0
    for j in range(0, display_rows):
        for i in range(0, display_cols):
            if curr_ex > m:
                    break 
            max_val = np.max(np.abs(X[curr_ex, :]))
            dai = pad + (j - 1) * (example_height + pad) + np.arange(0, example_height)
            daj = pad + (i - 1) * (example_width + pad) + np.arange(0, example_width)
            resh = X[curr_ex, :].reshape([example_height, example_width]) / max_val
            display_array[np.ix_(dai, daj)] = resh
            curr_ex += 1
        if curr_ex > m:
                break 
 
    plt.imshow(display_array.T, cmap='gray')
    plt.clim([-1, 1])
    plt.show()
    return display_array
