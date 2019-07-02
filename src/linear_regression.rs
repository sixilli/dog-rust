// Constants for basic regression properties
const LEARNING_RATE: f64 = 1e-3;
const MAX_ITER: usize = 102;

fn step(length: &f64, theta: &Vec<f64>, data: &Vec<f64>, target: &Vec<f64>) -> Vec<f64> {
    let mut preds = vec![] as Vec<f64>;
    let mut theta = vec![theta[0], theta[1]];

    for i in 0..data.len(){
        preds.push(theta[0] + theta[1] * data[i]);
    }

    let mut squared_diff = vec![] as Vec<f64>;
    for i in 0..preds.len(){
        squared_diff.push((preds[i] - target[i]).powi(2));
    }

    let squared_sum: f64 = squared_diff.iter().sum();

    let cost = (1.0 / (2.0 * length)) * squared_sum;

    //Updating theta parameters
    theta[0] = theta[0] - LEARNING_RATE * cost;
    theta[1] = theta[1] - LEARNING_RATE * cost;

    //Creating output vec to easily access data
    let output = vec![cost, theta[0], theta[1]];

    return output

}

pub fn linear_regression(data: &Vec<f64>, target: &Vec<f64>) -> Vec<f64> {
    let m = target.len() as f64;
    let mut coef = vec![1.0, 2.0];
    let mut iter = 0;
    let mut cost_hist: Vec<f64> = vec![];

    while iter < MAX_ITER {
        let output = step(&m, &coef, &data, &target);
        coef[0] = output[1];
        coef[1] = output[2];
        cost_hist.push(output[0]);

        if iter % 10 == 0 {
            println!("Iter: {} Cost: {} Theta 0: {} Theta1: {}", iter, output[0], output[1], output[2]);
            iter += 1;
        } else {
            iter += 1;
        }

    }
    return coef
}



fn main() {
    let data = vec![1f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let target = vec![2f64, 6.0, 4.0, 10.0, 5.0, 6.0];
    linear_regression(&data, &target);
}