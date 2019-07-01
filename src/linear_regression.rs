//pub struct LinearRegression {
//    coef: Vec<usize>,
//    coef_hist: Vec<usize>,
//    output: Vec<f32>,
//    iterations: usize,
//    done:  bool
//}
//
//impl LinearRegression {
//    fn get_base(data: &Vec<i32>, target: &Vec<i32>) -> LinearRegression {
//        RegressionBuilder::init().linear_regression(data, target)
//    }
//}

//const STEP_SIZE: f64 = 1e-2;
const LEARNING_RATE: f64 = 1e-3;
const MAX_ITER: usize = 2000;

pub struct RegressionBuilder {
    //lr: f64,
    //max_iter: usize
}

impl RegressionBuilder {
    // The core of the Linear Regression
    //pub fn init() -> RegressionBuilder {
    //    RegressionBuilder{
    //        lr: LEARNING_RATE,
    //        max_iter: MAX_ITER
    //    }
    //}

    //pub fn max_iter(self,  max_iter: usize) -> RegressionBuilder {
    //    RegressionBuilder { max_iter: max_iter, .. self}
    //}

    // Using gradient descent
    // Steps: Start with some theta parameters,
    //        Keep changing parameters to reduce cost
    pub fn linear_regression(data: &Vec<f64>, target: &Vec<f64>) {
        let m = target.len() as f64;
        let mut coef = vec![];
        let mut iter = 0;
        let mut cost_hist_w = vec![0.0];
        let mut cost_hist_b = vec![0.0];

        while iter < MAX_ITER {
            for (index, x_val) in data.iter().enumerate(){
                let mut x = x_val;
                let mut y = target[index];

                let mut bias = 0 as f64;
                let mut gradients = vec![] as Vec<f64>;
                // Allow for multivirate regression
                for grad in 0..coef.len() {
                    gradients[grad] = 0.0;
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
            }
        }
        println!("bias: {:?}, weight: {:?}", coef[0], coef[1]);
    }

}

fn main() {
    let data = vec![1f64, 2.0, 3.0];
    let target = vec![2f64, 6.0, 4.0];

    RegressionBuilder::linear_regression(&data, &target);
}