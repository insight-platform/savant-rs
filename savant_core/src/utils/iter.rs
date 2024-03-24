use std::ops::ControlFlow;

pub fn all_with_control_flow<I, F>(iter: I, mut f: F) -> ControlFlow<bool, bool>
where
    I: IntoIterator,
    F: FnMut(&I::Item) -> ControlFlow<bool, bool>,
{
    for item in iter {
        let res = f(&item);
        match &res {
            ControlFlow::Continue(true) => continue,
            _ => return res,
        }
    }
    ControlFlow::Continue(true)
}

pub fn any_with_control_flow<I, F>(iter: I, mut f: F) -> ControlFlow<bool, bool>
where
    I: IntoIterator,
    F: FnMut(&I::Item) -> ControlFlow<bool, bool>,
{
    for item in iter {
        let res = f(&item);
        match &res {
            ControlFlow::Continue(false) => continue,
            _ => return res,
        }
    }
    ControlFlow::Continue(false)
}

pub fn fiter_map_with_control_flow<I, F, T>(iter: I, mut f: F) -> Vec<T>
where
    I: IntoIterator<Item = T>,
    F: FnMut(&I::Item) -> ControlFlow<bool, bool>,
{
    let mut res = Vec::new();
    for item in iter {
        let r = f(&item);
        match r {
            ControlFlow::Continue(true) => res.push(item),
            ControlFlow::Break(v) => {
                if v {
                    res.push(item);
                }
                break;
            }
            _ => continue,
        }
    }
    res
}

pub fn partition_with_control_flow<I, F, T>(iter: I, mut f: F) -> (Vec<T>, Vec<T>)
where
    I: IntoIterator<Item = T>,
    F: FnMut(&I::Item) -> ControlFlow<bool, bool>,
{
    let mut first = Vec::new();
    let mut second = Vec::new();
    for item in iter {
        let r = f(&item);
        match r {
            ControlFlow::Continue(true) => first.push(item),
            ControlFlow::Continue(false) => second.push(item),
            ControlFlow::Break(v) => {
                if v {
                    first.push(item);
                } else {
                    second.push(item);
                }
                break;
            }
        }
    }
    (first, second)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_with_control_flow_continue_true() {
        let mut counter = 0;
        let f = |x: &i64| {
            counter += 1;
            ControlFlow::Continue(*x > 0)
        };
        let v = vec![1, 2, 3, 4, 5];
        let res = all_with_control_flow(v, f);
        assert_eq!(res, ControlFlow::Continue(true));
        assert_eq!(counter, 5);
    }

    #[test]
    fn test_all_with_control_flow_continue_false() {
        let mut counter = 0;
        let f = |x: &i64| {
            counter += 1;
            ControlFlow::Continue(*x > 0)
        };
        let v = vec![1, 2, 0, 4, 5];
        let res = all_with_control_flow(v, f);
        assert_eq!(res, ControlFlow::Continue(false));
        assert_eq!(counter, 3);
    }

    #[test]
    fn test_all_with_control_flow_break_false() {
        let mut counter = 0;
        let f = |_: &i64| {
            counter += 1;
            ControlFlow::Break(false)
        };
        let v = vec![1, 2, 3, 4, 5];
        let res = all_with_control_flow(v, f);
        assert_eq!(res, ControlFlow::Break(false));
        assert_eq!(counter, 1);
    }

    #[test]
    fn test_all_with_control_flow_break_true() {
        let mut counter = 0;
        let f = |_: &i64| {
            counter += 1;
            ControlFlow::Break(true)
        };
        let v = vec![1, 2, 3, 4, 5];
        let res = all_with_control_flow(v, f);
        assert_eq!(res, ControlFlow::Break(true));
        assert_eq!(counter, 1);
    }

    #[test]
    fn test_any_with_control_flow_continue_first_true() {
        let mut counter = 0;
        let f = |x: &i64| {
            counter += 1;
            ControlFlow::Continue(*x > 0)
        };
        let v = vec![1, 0, 0, 0, 0];
        let res = any_with_control_flow(v, f);
        assert_eq!(res, ControlFlow::Continue(true));
        assert_eq!(counter, 1);
    }

    #[test]
    fn test_any_with_control_flow_continue_last_true() {
        let mut counter = 0;
        let f = |x: &i64| {
            counter += 1;
            ControlFlow::Continue(*x > 0)
        };
        let v = vec![0, 0, 0, 0, 5];
        let res = any_with_control_flow(v, f);
        assert_eq!(res, ControlFlow::Continue(true));
        assert_eq!(counter, 5);
    }

    #[test]
    fn test_any_with_control_flow_continue_false() {
        let mut counter = 0;
        let f = |x: &i64| {
            counter += 1;
            ControlFlow::Continue(*x > 0)
        };
        let v = vec![0, 0, 0, 0, 0];
        let res = any_with_control_flow(v, f);
        assert_eq!(res, ControlFlow::Continue(false));
        assert_eq!(counter, 5);
    }

    #[test]
    fn test_any_with_control_flow_break_true() {
        let mut counter = 0;
        let f = |_: &i64| {
            counter += 1;
            ControlFlow::Break(true)
        };
        let v = vec![0, 0, 0, 0, 0];
        let res = any_with_control_flow(v, f);
        assert_eq!(res, ControlFlow::Break(true));
        assert_eq!(counter, 1);
    }

    #[test]
    fn test_any_with_control_flow_break_false() {
        let mut counter = 0;
        let f = |_: &i64| {
            counter += 1;
            ControlFlow::Break(false)
        };
        let v = vec![0, 0, 0, 0, 0];
        let res = any_with_control_flow(v, f);
        assert_eq!(res, ControlFlow::Break(false));
        assert_eq!(counter, 1);
    }

    #[test]
    fn test_filter_with_control_flow_continue() {
        let mut counter = 0;
        let f = |x: &i64| {
            counter += 1;
            ControlFlow::Continue(*x > 0)
        };
        let v = vec![1, 0, 3, 0, 5];
        let res = fiter_map_with_control_flow(v, f);
        assert_eq!(res, vec![1, 3, 5]);
        assert_eq!(counter, 5);
    }

    #[test]
    fn test_filter_with_control_flow_break() {
        let mut counter = 0;
        let f = |x: &i64| {
            counter += 1;
            ControlFlow::Break(*x > 0)
        };
        let v = vec![1, 0, 3, 0, 5];
        let res = fiter_map_with_control_flow(v, f);
        assert_eq!(res, vec![1]);
        assert_eq!(counter, 1);
    }

    #[test]
    fn test_partition_with_control_flow_continue() {
        let mut counter = 0;
        let f = |x: &i64| {
            counter += 1;
            ControlFlow::Continue(*x > 0)
        };
        let v = vec![1, 0, 3, 0, 5];
        let (first, second) = partition_with_control_flow(v, f);
        assert_eq!(first, vec![1, 3, 5]);
        assert_eq!(second, vec![0, 0]);
        assert_eq!(counter, 5);
    }

    #[test]
    fn test_partition_with_control_flow_first_3_break_false() {
        let mut counter = 0;
        let f = |x: &i64| {
            counter += 1;
            if counter > 3 {
                ControlFlow::Break(false)
            } else {
                ControlFlow::Continue(*x > 0)
            }
        };
        let v = vec![1, 0, 3, 0, 5];
        let (first, second) = partition_with_control_flow(v, f);
        assert_eq!(first, vec![1, 3]);
        assert_eq!(second, vec![0, 0]);
        assert_eq!(counter, 4);
    }

    #[test]
    fn test_partition_with_control_flow_first_3_break_true() {
        let mut counter = 0;
        let f = |x: &i64| {
            counter += 1;
            if counter > 3 {
                ControlFlow::Break(true)
            } else {
                ControlFlow::Continue(*x > 0)
            }
        };
        let v = vec![1, 0, 3, 0, 5];
        let (first, second) = partition_with_control_flow(v, f);
        assert_eq!(first, vec![1, 3, 0]);
        assert_eq!(second, vec![0]);
        assert_eq!(counter, 4);
    }
}
