from PIL import Image
import numpy as np

from jwave import FourierSeries
from jwave.acoustics import simulate_wave_propagation
from jwave.geometry import Domain, Medium, TimeAxis, BLISensors

from jax import jit
from jax import numpy as jnp
import matplotlib.pyplot as plt

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf, bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf)
    return img


def half_scale(image):
    new_image = np.zeros(tuple(x // 2 for x in image.shape))
    new_image += image[::2, ::2, ::2]
    new_image += image[1::2, ::2, ::2]
    new_image += image[::2, 1::2, ::2]
    new_image += image[::2, ::2, 1::2]
    new_image += image[1::2, 1::2, ::2]
    new_image += image[1::2, ::2, 1::2]
    new_image += image[::2, 1::2, 1::2]
    new_image += image[1::2, 1::2, 1::2]
    return new_image


def mcx_to_real_coordinates(x, y, z, r0, dx):
    # print(x, y, z, r0, dx)
    return [r + (x+0.5) * dx for x, r in zip([x,y,z], r0)]

def real_to_jwave_coordinates(x, y, z, domain, cz_j):
    x, y, z = [a.copy() for a in [x,y,z]]
    z -= cz_j
    N = domain.N
    dx = domain.dx
    # print(x, y, z, N, dx)
    return [x/dx + (n-1) / 2 for x, dx, n in zip([x,y,z],dx, N)]

def simulate_acoustic(p0_data, p0_raw, domain: Domain, medium: Medium, time_axis: TimeAxis, geometry, cz_j=0):
    # cz = Center of simulation domain in metres.
    p0_values = np.zeros(domain.N)
    # cz_j = 10e-3
    
    # Find the location in the jwave field of view covered by the mcx simulation.
    range_max = real_to_jwave_coordinates(*[a * 1e-3 for a in mcx_to_real_coordinates(*(p0_data["n"] - 0.5), p0_data["r0"], p0_data["dx_mm"])], domain, cz_j=cz_j)
    range_min = real_to_jwave_coordinates(*[a * 1e-3 for a in mcx_to_real_coordinates(*(p0_data["n"]*0 - 0.5), p0_data["r0"], p0_data["dx_mm"])], domain, cz_j=cz_j)
    # print(range_max, range_min)
    # assert 1 == 2
    # Convert the geometry into JWave pixel coordinates
    geom = real_to_jwave_coordinates(*geometry.T, domain, cz_j=cz_j)
    # Convert the sensors into j-wave class. 
    sensors_positions = tuple([jnp.array(x) for x in geom])
    sensors = BLISensors(sensors_positions, domain.N)
    
    # Put the MCX data (scaled down) into the jwave simulation domain.
    slicer = tuple([slice(int(a), int(b)) for a, b in zip(range_min, range_max)])
    print(range_min, range_max)
    halved = half_scale(half_scale(p0_raw))
    print(p0_raw.shape, halved.shape, p0_values.shape)
    p0_values[slicer] = halved
    
    # Convert the initial pressure into a Fourier Series for JWave
    p0 = FourierSeries(jnp.expand_dims(jnp.array(p0_values), -1), domain)
    plt.imshow(np.sum(np.squeeze(p0_values), axis=1))
    plt.show()
    # Get the GPU-compiled simulator.
    @jit
    def compiled_simulator(medium, p0):
        a = simulate_wave_propagation(medium, time_axis, p0=p0, sensors=sensors)
        return a

    sensors_data = compiled_simulator(medium, p0)[..., 0]
    return np.array(sensors_data)
