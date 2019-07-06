// Big thanks to /u/leudz for feedback on properly utilizng Rust
// Run time improved from 15ms to 2ms
// Constants for basic regression properties

const LEARNING_RATE: f64 = 1e-3;
const MAX_ITER: usize = 1000;

fn step([mut theta0, mut theta1]: [f64; 2], data: &[f64], target: &[f64]) -> [f64; 3] {
    let mut sum_reg_diff = 0.0;
    let mut x_sum_reg_diff = 0.0;

    let squared_sum: f64 = data
        .iter()
        .zip(target.iter())
        .map(|(&data, &target)| {
            let pred = theta0 + theta1 * data;

            sum_reg_diff += pred - target;
            x_sum_reg_diff += (pred - target) * data;

            (pred - target).powi(2)
        })
        .sum();

    let length = target.len() as f64;
    let loss = (1.0 / (2.0 * length)) * squared_sum;

    theta0 -= LEARNING_RATE * (1.0 / length) * sum_reg_diff;
    theta1 -= LEARNING_RATE * (1.0 / length) * x_sum_reg_diff;

    [loss, theta0, theta1]
}

pub fn linear_regression(data: &[f64], target: &[f64]) -> Vec<f64> {
    let mut coef = vec![1.0, 2.0];
    let mut loss_hist = Vec::with_capacity(MAX_ITER);

    for iter in 0..=MAX_ITER {
        let [loss, theta0, theta1] = step([coef[0], coef[1]], &data, &target);

        // Updating coefficients and apending to loss history
        coef[0] = theta0;
        coef[1] = theta1;
        loss_hist.push(loss);

        // For debugging purposes, print some metrics every 100 steps
        if iter % 100 == 0 {
            println!(
                "Iter: {} Cost: {} Theta 0: {} Theta1: {}",
                iter, loss, theta0, theta1
            );
        }
    }
    coef
}