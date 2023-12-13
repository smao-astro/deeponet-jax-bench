"""
Training with my best knowledge of JAX
"""

import argparse
from functools import partial
from time import time

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from tqdm import trange

from src.model import DeepONet
from src.utils import mse_to_zeros, batched_l2_relative_error, DiffRecData


def get_jac_fn(fwd_fn):
    @partial(jax.vmap, in_axes=(0, None))
    @partial(jax.vmap, in_axes=(None, 0))
    def f(branch_input_, trunk_input_):
        """

        Args:
            branch_input_:
            trunk_input_:

        Returns:
            shape: (M, N, 2)
        """
        return jax.grad(fwd_fn, argnums=1)(branch_input_, trunk_input_)
    return f


def get_hessian_fn(fwd_fn):
    @partial(jax.vmap, in_axes=(0, None))
    @partial(jax.vmap, in_axes=(None, 0))
    def f(branch_input_, trunk_input_):
        """

        Args:
            branch_input_:
            trunk_input_:

        Returns:
            shape: (M, N, 2, 2)
        """
        return jax.jacfwd(jax.grad(fwd_fn, argnums=1), argnums=1)(branch_input_, trunk_input_)
    return f

def compute_u_pde(forward_fn, branch_input, trunk_input, source_input):
    """ diffusion-reaction equation with VMAP over function dimension """
    # constants
    d, k = 0.01, 0.01

    u = forward_fn(branch_input, trunk_input)
    du = get_jac_fn(forward_fn)(branch_input, trunk_input)
    ddu = get_hessian_fn(forward_fn)(branch_input, trunk_input)

    # pde
    pde = du[:, :, 1] - d * ddu[:, :, 0, 0] + k * u ** 2 - source_input

    return u, pde


@partial(jax.jit, static_argnames=['n_points_pde', 'n_points_bc', 'model_interface'])
def train_step(state, branch_input, trunk_input, source_input,
               n_points_pde, n_points_bc, model_interface):
    """ train for a single step """

    def loss_fn(params_):
        """ loss function, for AD w.r.t. network weights """

        def forward_fn(branch_input_, trunk_input_):
            """ forward function, for AD w.r.t. coordinates """
            return model_interface.apply({'params': params_},
                                         branch_in=branch_input_, trunk_in=trunk_input_)

        u_val, pde_val = compute_u_pde(forward_fn, branch_input, trunk_input, source_input)
        pde_loss_ = mse_to_zeros(pde_val[:, :n_points_pde])
        bc_loss_ = mse_to_zeros(u_val[:, n_points_pde:n_points_pde + n_points_bc])
        ic_loss_ = mse_to_zeros(u_val[:, n_points_pde + n_points_bc:])
        return pde_loss_ + bc_loss_ + ic_loss_, (pde_loss_, bc_loss_, ic_loss_)

    # loss
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (pde_loss, bc_loss, ic_loss)), grads = grad_fn(state.params)
    # update
    state = state.apply_gradients(grads=grads)
    return state, loss, pde_loss, bc_loss, ic_loss


def run():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--iterations', type=int, default=1000,
                        help='number of iterations')
    parser.add_argument('-M', '--n-functions', type=int, default=50,
                        help='number of functions in a batch')
    parser.add_argument('-N', '--n-points', type=int, default=4000,
                        help='number of collocation points in a batch')
    args = parser.parse_args()

    # load data
    data = DiffRecData(data_path='./data')

    # number of functions and points in a batch
    # Note: the default values come from the diff_rec example in DeepXDE-ZCS
    #       for comparison with pytorch, tensorflow and paddle
    N_FUNCTIONS = args.n_functions  # noqa
    N_POINTS_PDE = args.n_points  # noqa

    # train state
    model = DeepONet(branch_features=(data.n_features, 128, 128, 128),
                     trunk_features=(data.n_dims, 128, 128, 128),
                     cartesian_prod=True)
    branch_in, trunk_in, _, _ = data.sample_batch(N_FUNCTIONS, N_POINTS_PDE)
    params = model.init(jax.random.PRNGKey(0), branch_in, trunk_in)['params']
    optimizer = optax.adam(learning_rate=0.0005)
    the_state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optimizer)

    #################
    # training loop #
    #################
    pbar = trange(args.iterations, desc='Training')
    t_total = 0.
    for it in pbar:
        # sample data
        branch_in, trunk_in, source_in, _ = data.sample_batch(
            N_FUNCTIONS, N_POINTS_PDE, seed=it, train=True)

        # update
        t0 = time()  # wall time excluding data sampling
        the_state, loss_val, pde_loss_val, bc_loss_val, ic_loss_val = \
            train_step(the_state, branch_in, trunk_in, source_in,
                       N_POINTS_PDE, data.n_bc, model)

        pbar.set_postfix_str(f"L_pde={pde_loss_val:.4e}, "
                             f"L_bc={bc_loss_val:.4e}, "
                             f"L_ic={ic_loss_val:.4e}, "
                             f"L={loss_val:.4e}")
        t_total += time() - t0
    print(f'Training done in {t_total:.1f} seconds')

    ##############
    # evaluation #
    ##############
    # sample solution
    print('Validating against true solution...')
    n_functions_eval, n_points_eval = 200, 10000
    branch_in, trunk_in, source_in, u_true = data.sample_batch(
        n_functions_eval, n_points_eval, train=False)
    # prediction
    u_pred = model.apply({'params': the_state.params}, branch_in=branch_in, trunk_in=trunk_in)
    print(f'L2 relative error: {batched_l2_relative_error(u_true, u_pred)}')


if __name__ == "__main__":
    run()
