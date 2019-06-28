pub struct LinearRegression<T> {
    coef: Vec<usize>,
    output: Vec<f32>,
    iterations: usize,
    done:  bool
}

impl<T> LinearRegression<T> {
    fn get_base(data: &Vec<T>, target: &Vec<T>) -> LinearRegression<T> {
        RegressionBuilder::init().linear_regression(data, target)
    }

    fn sgd(&self) -> Vec<(Vec<usize>, Vec<T>)> {

    }
}

const BETA: f64 = 1e-6;
const LEARNING_RATE: f64 = 1e-6;
const MAX_ITER: usize = 100;

pub struct RegressionBuilder {
    beta: f64,
    lr: f64,
    max_iter: usize
}

impl RegressionBuilder {
    // The core of the Linear Regression
    pub fn init() -> RegressionBuilder {
        RegressionBuilder{
            beta: BETA,
            lr: LEARNING_RATE,
            max_iter: MAX_ITER
        }
    }

    pub fn max_iter(self,  max_iter: usize) -> RegressionBuilder {
        RegressionBuilder { max_iter: max_iter, .. self}
    }

    pub fn linear_regression(self, data: &Vec<T>, target: &Vec<T>) {

    }
}