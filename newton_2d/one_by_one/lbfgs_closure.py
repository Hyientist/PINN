import torch


def lbfgs_closure(dataloader_, optimizer_, model_, equation_, dt_):
    """
    Creates a closure for use with the LBFGS optimizer in PyTorch. This closure
    computes the loss for a batch of data, performs backpropagation to calculate
    the gradients, and returns the total loss.

    Parameters:
    dataloader_ (torch.utils.data.DataLoader): The data loader providing batches
                                               of input data and labels.
    optimizer_ (torch.optim.Optimizer): The optimizer used for updating the model
                                        parameters. Typically an instance of LBFGS.
    model_ (torch.nn.Module): The neural network model that predicts outputs
                              from the input data.
    equation_ (callable): A function that takes model outputs and other parameters
                          to compute additional constraints or physical equations
                          relevant to the problem.
    dt_ (torch.Tensor or float): The time step used in the physical equation
                                 or the model update.

    Returns:
    callable: A closure function that computes the total loss for the current
              batch of data, performs backpropagation, and returns the total loss.

    Details:
    - The closure function is designed to be used with the `LBFGS` optimizer in PyTorch,
      which requires a closure function to be passed to its `step` method.
    - The closure resets the gradients using `optimizer_.zero_grad()` at the start of
      each iteration to ensure that gradients are not accumulated across batches.
    - For each batch of data provided by `dataloader_`, the model predictions are
      obtained by passing `input_chunk` through the model. The first four columns
      of the model output (x, y, u, v) are extracted, and the observed loss is
      computed using the mean squared error between the model predictions and
      the true labels.
    - The mean value of the fifth column, `model_cd`, is extracted from the model
      output, representing a parameter related to drag or another physical property.
    - The physical consistency loss (`loss_eqn`) is computed by passing the model
      outputs (x, y, u, v), the drag coefficient, and the time step `dt_` to the
      `equation_` function, which returns expected results based on the physical
      model. The loss is then computed as the mean squared error between these
      results and the true labels.
    - The total loss is the sum of the observed loss (`loss_obs`) and the physical
      consistency loss (`loss_eqn`). This total loss is used to perform backpropagation.

    CAUTION:
    - The code contains sections that compute gradients with respect to the input
      (`grad_x`, `grad_y`) and additional loss terms (`loss_u`, `loss_v`).
      Uncommenting these lines will add gradient-based constraints to the total loss
      function.

      WARNING: Including these additional loss terms can significantly alter the
      learning dynamics of the model. In particular, the predicted value of `cd`
      (drag coefficient) may change, as these terms enforce stricter physical
      consistency constraints. It is essential to understand the impact of these
      additional losses on the overall training process. To understand this impact:

      1. **Run Training Without `loss_u` and `loss_v`**: Comment out the additional
         loss terms and train the model. Observe the predicted `cd` value.

      2. **Run Training With `loss_u` and `loss_v`**: Uncomment the additional loss
         terms, include them in the total loss, and train the model. Compare the
         predicted `cd` value with the previous step.

      3. **Compare Results**: Analyze how the inclusion of these gradient-based loss
         terms affects the `cd` value and overall model performance.

    Example Usage:
    ```
    optimizer = torch.optim.LBFGS(model_.parameters())
    closure = lbfgs_closure(dataloader_, optimizer_, model_, equation_, dt_)
    optimizer.step(closure)
    ```
    """

    def closure():
        optimizer_.zero_grad()  # Reset gradients to avoid accumulation
        total_loss = 0  # Initialize total loss
        lossfn = torch.nn.MSELoss()  # Define the loss function (Mean Squared Error)

        # Iterate over each batch of data from the dataloader
        for input_chunk, label_chunk in dataloader_:
            # Forward pass: compute the model output
            model_output_chunk = model_(input_chunk)
            # Extract the first four components (x, y, u, v) of the output
            model_output_xyuv = model_output_chunk[:, :4]
            # Compute the observed loss between the model output and the true labels
            loss_obs = lossfn(model_output_xyuv, label_chunk)

            # Extract individual components for further computation
            model_x = model_output_chunk[:, 0]
            model_y = model_output_chunk[:, 1]
            # model_u and model_v are the velocity components
            model_u = model_output_chunk[:, 2]
            model_v = model_output_chunk[:, 3]
            # model_cd is a drag coefficient or another physical parameter
            model_cd = torch.mean(model_output_chunk[:, 4])

            # Compute the equation loss based on physical consistency
            eqn_ans = equation_(model_x, model_y, model_u, model_v, model_cd, dt_)
            loss_eqn = lossfn(eqn_ans, label_chunk)

            # Combine observed loss and equation loss into total loss
            total_loss = loss_eqn + loss_obs

            # ------------------------- CAUTION -------------------------
            # Uncomment the following lines to include additional gradient-based losses
            # WARNING: Including these additional losses will change the predicted `cd` value.
            # grad_x = torch.autograd.grad(
            #     model_x, input_chunk,
            #     grad_outputs=torch.ones_like(model_x),
            #     retain_graph=True,
            #     create_graph=True
            # )[0].squeeze()
            # grad_y = torch.autograd.grad(
            #     model_y, input_chunk,
            #     grad_outputs=torch.ones_like(model_y),
            #     retain_graph=True,
            #     create_graph=True
            # )[0].squeeze()
            # loss_u = lossfn(grad_x, model_u)
            # loss_v = lossfn(grad_y, model_v)
            # total_loss = loss_eqn + loss_obs + loss_u + loss_v
            # -----------------------------------------------------------

            # Print debugging information for the current batch
            print("----------------------------")
            print("Value of Cd:", model_cd.item())
            print("loss of obs:", loss_obs.item())
            print("loss of eqn:", loss_eqn.item())
            # print("loss of u:", loss_u.item())
            # print("loss of v:", loss_v.item())
            print("")

        total_loss.backward()  # Compute gradients via backpropagation
        return total_loss  # Return the total loss for the optimizer

    return closure  # Return the closure function for the optimizer

