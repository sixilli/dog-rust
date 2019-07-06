use std::time::Instant;

mod linear_regression;
use linear_regression::linear_regression;

fn main() {
    let now = Instant::now();
    let data = vec![1f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let target = vec![2f64, 6.0, 4.0, 10.0, 5.0, 6.0];
    linear_regression(&data, &target);
    let new_now = Instant::now();
    println!("{:?}", new_now.duration_since(now));
}