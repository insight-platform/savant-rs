#[cfg(test)]
mod tests {
    use rkyv::{Archive, Deserialize, Serialize};

    #[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
    #[archive(check_bytes)]
    pub enum Variant {
        Tensor(Vec<i64>, Vec<u8>),
        Integer(i64),
        Float(f64),
    }

    #[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone, Default)]
    #[archive(check_bytes)]
    pub struct Outer {
        values: Vec<Variant>,
    }

    #[test]
    fn test_ser_des_variant() {
        let mut outer = Outer::default();
        outer.values.push(Variant::Integer(1));
        outer.values.push(Variant::Float(2.0));
        outer
            .values
            .push(Variant::Tensor(vec![8, 3, 8, 8], [0; 3 * 8 * 8].to_vec()));

        let mut bytes = Vec::with_capacity(760);
        bytes.extend_from_slice(
            rkyv::to_bytes::<_, 756>(&outer)
                .expect("Failed to serialize VideoFrame")
                .as_ref(),
        );

        let o: Result<Outer, _> = rkyv::from_bytes(&bytes[..]);
        assert!(o.is_ok());
    }
}
