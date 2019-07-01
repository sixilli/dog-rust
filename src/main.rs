// Constants for basic regression properties
const LEARNING_RATE: f64 = 1e-3;
const MAX_ITER: usize = 20;

fn cost_func(length: &f64, theta: &Vec<f64>, data: &Vec<f64>, target: &Vec<f64>) -> f64 {
    let preds = vec![] as Vec<f64>;
    for i in 0..data.len(){
        preds.push(theta[0] + theta[1] * data[i]);
    }

    let squared_diff = vec![] as Vec<f64>;
    for i in 0..preds.len(){
        squared_diff.push((preds[i] - target[i]).powi(2));
    }

    let cost = (1.0 / (2.0 * length)) * squared_diff.iter().sum();
    return cost

}

fn gradient_descent(theta: &Vec<f64>, data: &Vec<f64>, target: &Vec<f64>) -> Vec<f64> {
    theta;
}

pub fn linear_regression(data: &Vec<f64>, target: &Vec<f64>) -> Vec<f64> {
    let m = target.len() as f64;
    let mut coef = vec![1.0, 2.0];
    let mut iter = 0;
    let mut cost_hist_w = vec![];
    let mut cost_hist_b = vec![];

    while iter < MAX_ITER {
        for (x_val, y_val) in data.iter().zip(target.iter()) {
            let x = x_val;
            let y = y_val;
            let mut bias = 0 as f64;
            let mut gradients = vec![] as Vec<f64>;
            // Allow for multivirate regression
            for g in 0..coef.len() {
                gradients.push(0.0);
            }

            // Performing Gradient Descent
            // Calculating Gradient
            for g in 0..gradients.len() {
                gradients[g] += -2.0 * x * (y - (gradients[g] + bias));
                bias += -2.0 * (y - (gradients[g] + bias));
            }

            // Updating Gradient
            bias -= (bias / m) * LEARNING_RATE;
            for g in 0..gradients.len() {
                gradients[g] -= (gradients[g] / m) as f64 * LEARNING_RATE;
            }

            for g in 0..gradients.len() {
                if g == 0 {
                    coef[g] = bias;
                } else{
                    coef[g] = gradients[g];
                }
            }
            cost_hist_w.push(coef[1]);
            cost_hist_b.push(coef[0]);
            iter += 1;
            println!("bias {} weight {}", coef[0], coef[1])
        }
    }
    return coef
}



fn main() {
    let data = vec![1f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let target = vec![2f64, 6.0, 4.0, 10.0, 5.0, 6.0];
    let reg = linear_regression(&data, &target);
    println!("bias: {}, weight: {}", reg[0], reg[1]);
}