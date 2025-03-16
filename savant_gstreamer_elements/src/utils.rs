const DEFAULT_TIME_BASE: (i32, i32) = (1, 1_000_000_000);

pub fn convert_ts(ts: i64, time_base: (i32, i32)) -> i64 {
    if time_base == DEFAULT_TIME_BASE {
        ts
    } else {
        let (tb_num, tb_denum) = time_base;
        let (target_num, target_denum) = DEFAULT_TIME_BASE;
        (ts * target_num as i64 * tb_denum as i64) / (target_denum as i64 * tb_num as i64)
    }
}
