// Constants for basic regression properties
const LEARNING_RATE: f64 = 1e-3;
const MAX_ITER: usize = 1001;

fn step(length: &f64, theta: &Vec<f64>, data: &Vec<f64>, target: &Vec<f64>) -> Vec<f64> {
    // Intializing vectors for data.
    let mut preds = vec![] as Vec<f64>;
    let mut theta = vec![theta[0], theta[1]];

    // Getting predictions
    for i in 0..data.len(){
        preds.push(theta[0] + theta[1] * data[i]);
    }

    // Using squared difference to calculate loss.
    let mut squared_diff = vec![] as Vec<f64>;
    for i in 0..preds.len(){
        squared_diff.push((preds[i] - target[i]).powi(2));
    }

    // Sum of total loss
    let squared_sum: f64 = squared_diff.iter().sum();

    // Finding average loss
    let loss = (1.0 / (2.0 * length)) * squared_sum;

    // Finding difference between prediction and target
    // Creating two different vecs as theta0 and theta1
    // Require different approaches
    let mut reg_diff = vec![] as Vec<f64>;
    let mut x_reg_diff = vec! [] as Vec<f64>;

    for i in 0..preds.len(){
        reg_diff.push(preds[i] - target[i]);
        x_reg_diff.push((preds[i] - target[i]) * data[i]);
    }

    let sum_reg_diff: f64 = reg_diff.iter().sum();
    let x_sum_reg_diff: f64 = x_reg_diff.iter().sum();

    //Applying gradient descent to theta parameters
    theta[0] = theta[0] - LEARNING_RATE  * (1.0 / length) * sum_reg_diff;
    theta[1] = theta[1] - LEARNING_RATE  * (1.0 / length) * x_sum_reg_diff;

    //Creating output vec to easily access data
    let output = vec![loss, theta[0], theta[1]];

    return output

}

pub fn linear_regression(data: &Vec<f64>, target: &Vec<f64>) -> Vec<f64> {
    // Creating parameters for running regression
    let m = target.len() as f64;
    let mut coef = vec![1.0, 2.0];
    let mut iter = 0;
    let mut loss_hist: Vec<f64> = vec![];

    // Running model until max iter is reached
    while iter < MAX_ITER {
        let output = step(&m, &coef, &data, &target);

        // Updating coefficients and apending to loss history
        coef[0] = output[1];
        coef[1] = output[2];
        loss_hist.push(output[0]);

        // For debugging purposes, print some metrics every 10 steps
        if iter % 100 == 0 {
            println!("Iter: {} Cost: {} Theta 0: {} Theta1: {}", iter, output[0], output[1], output[2]);
            iter += 1;
        } else {
            iter += 1;
        }

    }
    return coef
}