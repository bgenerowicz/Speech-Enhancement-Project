import matplotlib.pyplot as plt
import pylab

def make_plot(data,reconstructed_data):
    # Calculate Residual
    residual = data - reconstructed_data

    # Plots
    f, axarr = plt.subplots(3, sharex=True)
    axarr[2].plot(residual)
    axarr[2].set_title('Residual')
    pylab.ylim([-0.5, 0.5])
    axarr[1].plot(reconstructed_data)
    axarr[1].set_title('Reconstructed')
    axarr[0].plot(data)
    axarr[0].set_title('Original')

    plt.show()