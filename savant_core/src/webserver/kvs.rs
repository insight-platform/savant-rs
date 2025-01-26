pub mod asynchronous {
    use crate::primitives::attribute::Attribute;
    use crate::webserver::{KvsOperation, KvsOperationKind, WsData, WS_DATA};
    use globset::Glob;
    use std::time::SystemTime;

    pub async fn set_attributes(attributes: &[Attribute], ttl: Option<u64>) {
        for attr in attributes {
            let namespace = attr.namespace.clone();
            let name = attr.name.clone();
            WS_DATA
                .kvs
                .insert((namespace.clone(), name.clone()), (ttl, attr.clone()))
                .await;
        }
        let subscribers = WS_DATA.kvs_subscribers.clone();
        WsData::broadcast_kvs_operation(
            subscribers,
            KvsOperation {
                timestamp: SystemTime::now(),
                operation: KvsOperationKind::Set(attributes.to_vec(), ttl),
            },
        )
        .await;
    }

    pub async fn search_attributes(ns: &Option<String>, name: &Option<String>) -> Vec<Attribute> {
        let ns_glob = ns
            .as_ref()
            .map(|s| Glob::new(s.as_str()))
            .unwrap_or(Glob::new("*"))
            .unwrap()
            .compile_matcher();

        let name_glob = name
            .as_ref()
            .map(|s| Glob::new(s.as_str()))
            .unwrap_or(Glob::new("*"))
            .unwrap()
            .compile_matcher();

        let mut attr_set = Vec::new();
        for (key, (_, attr)) in WS_DATA.kvs.iter() {
            let key_ns = &key.0;
            let key_name = &key.1;
            if ns_glob.is_match(key_ns) && name_glob.is_match(key_name) {
                attr_set.push(attr.clone());
            }
        }
        attr_set
    }

    pub async fn search_keys(ns: &Option<String>, name: &Option<String>) -> Vec<(String, String)> {
        let mut keys = Vec::new();
        let ns_glob = ns
            .as_ref()
            .map(|s| Glob::new(s.as_str()))
            .unwrap_or(Glob::new("*"))
            .unwrap()
            .compile_matcher();

        let name_glob = name
            .as_ref()
            .map(|s| Glob::new(s.as_str()))
            .unwrap_or(Glob::new("*"))
            .unwrap()
            .compile_matcher();

        for (key, _) in WS_DATA.kvs.iter() {
            let key_ns = &key.0;
            let key_name = &key.1;
            if ns_glob.is_match(key_ns) && name_glob.is_match(key_name) {
                keys.push((key_ns.clone(), key_name.clone()));
            }
        }
        keys
    }

    pub async fn del_attributes(ns: &Option<String>, name: &Option<String>) {
        let mut keys_to_delete = Vec::new();
        let ns_glob = ns
            .as_ref()
            .map(|s| Glob::new(s.as_str()))
            .unwrap_or(Glob::new("*"))
            .unwrap()
            .compile_matcher();

        let name_glob = name
            .as_ref()
            .map(|s| Glob::new(s.as_str()))
            .unwrap_or(Glob::new("*"))
            .unwrap()
            .compile_matcher();

        for (key, _) in WS_DATA.kvs.iter() {
            let key_ns = &key.0;
            let key_name = &key.1;
            if ns_glob.is_match(key_ns) && name_glob.is_match(key_name) {
                keys_to_delete.push(key.clone());
            }
        }
        let subscribers = WS_DATA.kvs_subscribers.clone();
        let mut attrs = Vec::with_capacity(keys_to_delete.len());
        for key in keys_to_delete {
            let res = WS_DATA.kvs.remove(&key).await;
            if let Some((_ttl, attr)) = res {
                attrs.push(attr);
            }
        }
        WsData::broadcast_kvs_operation(
            subscribers.clone(),
            KvsOperation {
                timestamp: SystemTime::now(),
                operation: KvsOperationKind::Delete(attrs),
            },
        )
        .await;
    }

    pub async fn get_attribute(ns: &str, name: &str) -> Option<Attribute> {
        WS_DATA
            .kvs
            .get(&(ns.to_string(), name.to_string()))
            .await
            .map(|(_, attr)| attr)
    }

    pub async fn del_attribute(ns: &str, name: &str) -> Option<Attribute> {
        let res = WS_DATA
            .kvs
            .remove(&(ns.to_string(), name.to_string()))
            .await
            .map(|(_, attr)| attr);
        if let Some(attr) = res {
            let subscribers = WS_DATA.kvs_subscribers.clone();
            WsData::broadcast_kvs_operation(
                subscribers,
                KvsOperation {
                    timestamp: SystemTime::now(),
                    operation: KvsOperationKind::Delete(vec![attr.clone()]),
                },
            )
            .await;
            Some(attr)
        } else {
            None
        }
    }
}

pub mod synchronous {
    use crate::get_or_init_async_runtime;
    use crate::primitives::attribute::Attribute;

    pub fn set_attributes(attributes: &[Attribute], ttl: Option<u64>) {
        let rt = get_or_init_async_runtime();
        rt.block_on(async {
            crate::webserver::kvs::asynchronous::set_attributes(attributes, ttl).await
        });
    }

    pub fn search_attributes(ns: &Option<String>, name: &Option<String>) -> Vec<Attribute> {
        let rt = get_or_init_async_runtime();
        rt.block_on(async {
            crate::webserver::kvs::asynchronous::search_attributes(ns, name).await
        })
    }

    pub fn search_keys(ns: &Option<String>, name: &Option<String>) -> Vec<(String, String)> {
        let rt = get_or_init_async_runtime();
        rt.block_on(async { crate::webserver::kvs::asynchronous::search_keys(ns, name).await })
    }

    pub fn del_attributes(ns: &Option<String>, name: &Option<String>) {
        let rt = get_or_init_async_runtime();
        rt.block_on(async { crate::webserver::kvs::asynchronous::del_attributes(ns, name).await });
    }

    pub fn get_attribute(ns: &str, name: &str) -> Option<Attribute> {
        let rt = get_or_init_async_runtime();
        rt.block_on(async { crate::webserver::kvs::asynchronous::get_attribute(ns, name).await })
    }

    pub fn del_attribute(ns: &str, name: &str) -> Option<Attribute> {
        let rt = get_or_init_async_runtime();
        rt.block_on(async { crate::webserver::kvs::asynchronous::del_attribute(ns, name).await })
    }
}

#[cfg(test)]
mod tests {
    use crate::get_or_init_async_runtime;
    use crate::primitives::attribute::Attribute;
    use crate::primitives::attribute_set::AttributeSet;
    use crate::primitives::attribute_value::AttributeValue;
    use crate::protobuf::{from_pb, ToProtobuf};
    use crate::webserver::kvs::synchronous::*;
    use crate::webserver::{
        init_webserver, set_status, stop_webserver, KvsOperation, KvsOperationKind, PipelineStatus,
        HOUSKEEPING_PERIOD, WS_DATA,
    };
    use savant_protobuf::generated;
    use std::thread::sleep;
    use std::time::Duration;

    #[test]
    #[serial_test::serial]
    fn test_kvs() {
        del_attributes(&None, &None);
        let attribute_set = vec![
            Attribute::persistent("abc", "xax", vec![], &None, false),
            Attribute::persistent("ghi", "yay", vec![], &None, false),
        ];
        set_attributes(&attribute_set, None);
        let retrieved_all = search_attributes(&None, &None);
        assert_eq!(retrieved_all.len(), 2);
        let retrieved_abc = search_attributes(&Some("abc".to_string()), &None);
        assert_eq!(retrieved_abc.len(), 1);
        let retrieved_with_glob = search_attributes(&None, &Some("?a?".to_string()));
        assert_eq!(retrieved_with_glob.len(), 2);

        let ttl_attribute_set = vec![
            Attribute::persistent("def", "xax", vec![], &None, false),
            Attribute::persistent("jkl", "yay", vec![], &None, false),
        ];

        set_attributes(&ttl_attribute_set, Some(10));
        let retrieved_all = search_attributes(&None, &None);
        assert_eq!(retrieved_all.len(), 4);
        sleep(std::time::Duration::from_millis(11));
        let retrieved_all = search_attributes(&None, &None);
        assert_eq!(retrieved_all.len(), 2);

        let abc_attribute = get_attribute(&"abc".to_string(), &"xax".to_string());
        assert_eq!(abc_attribute.as_ref().unwrap().name.as_str(), "xax");

        del_attributes(&None, &None);
        let retrieved_all = search_attributes(&None, &None);
        assert_eq!(retrieved_all.len(), 0);
    }

    #[test]
    #[serial_test::serial]
    fn test_replace_kv() {
        del_attributes(&None, &None);
        let old_attr = vec![Attribute::persistent(
            "abc",
            "xax",
            vec![AttributeValue::integer(1, None)],
            &None,
            false,
        )];
        let new_attr = vec![Attribute::persistent(
            "abc",
            "xax",
            vec![AttributeValue::integer(2, None)],
            &None,
            false,
        )];
        set_attributes(&old_attr, None);
        set_attributes(&new_attr, None);
        let retrieved_all = search_attributes(&None, &None);
        assert_eq!(retrieved_all.len(), 1);
        let retrieved_attr = &retrieved_all[0];
        assert_eq!(&retrieved_attr.values[0], &new_attr[0].values[0]);
        del_attributes(&None, &None);
    }

    #[test]
    #[serial_test::serial]
    fn test_subscription() -> anyhow::Result<()> {
        del_attributes(&None, &None);
        sleep(HOUSKEEPING_PERIOD + Duration::from_millis(10)); // wait until subscription handler clear the op queue
        let rt = get_or_init_async_runtime();
        let mut subscription = rt.block_on(async { WS_DATA.subscribe("me", 10).await })?;

        let ttl_attribute_set = vec![Attribute::persistent("jkl", "yay", vec![], &None, false)];
        let attribute_set = vec![Attribute::persistent("ghi", "yay", vec![], &None, false)];
        set_attributes(&ttl_attribute_set, Some(200));
        set_attributes(&attribute_set, None);
        let received = rt.block_on(async { subscription.recv().await }).unwrap();
        assert!(matches!(
            received,
            KvsOperation {
                timestamp: _,
                operation: KvsOperationKind::Set(attr, ttl)
            } if attr[0].namespace.as_str() == "jkl" && attr[0].name.as_str() == "yay" && ttl == Some(200)
        ));
        let received = rt.block_on(async { subscription.recv().await }).unwrap();
        assert!(matches!(
            received,
            KvsOperation {
                timestamp: _,
                operation: KvsOperationKind::Set(attr, ttl)
            } if attr[0].namespace.as_str() == "ghi" && attr[0].name.as_str() == "yay" && ttl.is_none()
        ));
        del_attribute("ghi", "yay");
        let received = rt.block_on(async { subscription.recv().await }).unwrap();
        assert!(matches!(
            received,
            KvsOperation {
                timestamp: _,
                operation: KvsOperationKind::Delete(attr)
            } if attr[0].namespace.as_str() == "ghi" && attr[0].name.as_str() == "yay"
        ));
        set_attributes(&attribute_set, None);
        let received = rt.block_on(async { subscription.recv().await }).unwrap();
        assert!(matches!(
            received,
            KvsOperation {
                timestamp: _,
                operation: KvsOperationKind::Set(attr, ttl)
            } if attr[0].namespace.as_str() == "ghi" && attr[0].name.as_str() == "yay" && ttl.is_none()
        ));
        Ok(())
    }

    #[test]
    #[serial_test::serial]
    fn test_abi_to_api() -> anyhow::Result<()> {
        del_attributes(&None, &None);
        init_webserver(8888)?;
        sleep(Duration::from_millis(100));
        set_status(PipelineStatus::Running)?;
        let ttl_attribute_set = vec![Attribute::persistent("jkl", "yay", vec![], &None, false)];
        let attribute_set = vec![Attribute::persistent("ghi", "yay", vec![], &None, false)];
        set_attributes(&ttl_attribute_set, Some(2000));
        set_attributes(&attribute_set, None);

        let r = reqwest::blocking::get("http://localhost:8888/kvs/search-keys/*/*")?;
        assert_eq!(r.status(), 200);
        let mut result: Vec<(String, String)> = r.json()?;
        result.sort_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        assert_eq!(
            result,
            vec![
                ("ghi".to_string(), "yay".to_string()),
                ("jkl".to_string(), "yay".to_string())
            ]
        );
        sleep(Duration::from_millis(2001));

        let r = reqwest::blocking::get("http://localhost:8888/kvs/search-keys/*/*")?;
        assert_eq!(r.status(), 200);
        let result: Vec<(String, String)> = r.json()?;
        assert_eq!(result, vec![("ghi".to_string(), "yay".to_string())]);

        let r = reqwest::blocking::get("http://localhost:8888/kvs/search/*/*")?;
        assert_eq!(r.status(), 200);
        let binary = r.bytes()?;
        let res_attribute_set = from_pb::<generated::AttributeSet, AttributeSet>(&binary)?;
        assert_eq!(res_attribute_set.attributes, attribute_set);

        let r = reqwest::blocking::get("http://localhost:8888/kvs/get/ghi/yay")?;
        assert_eq!(r.status(), 200);
        let binary = r.bytes()?;
        let res_attribute = from_pb::<generated::AttributeSet, AttributeSet>(&binary)?.attributes;
        assert_eq!(res_attribute[0], attribute_set[0]);

        let rt = get_or_init_async_runtime();
        let client = reqwest::Client::new();
        // delete single
        let r = rt.block_on(async {
            let resp = client
                .post("http://localhost:8888/kvs/delete-single/ghi/yay")
                .send()
                .await
                .unwrap();
            assert_eq!(resp.status(), 200);
            resp.bytes().await
        })?;
        let res_attribute = from_pb::<generated::AttributeSet, AttributeSet>(&r)?.attributes;
        assert_eq!(res_attribute[0], attribute_set[0]);

        // delete after delete
        let r = rt.block_on(async {
            let resp = client
                .post("http://localhost:8888/kvs/delete-single/ghi/yay")
                .send()
                .await
                .unwrap();
            assert_eq!(resp.status(), 200);
            resp.bytes().await
        })?;
        let res_attribute = from_pb::<generated::AttributeSet, AttributeSet>(&r)?.attributes;
        assert_eq!(res_attribute.len(), 0);

        // set again and purge
        set_attributes(&attribute_set, None);
        let r = rt.block_on(async {
            let resp = client
                .post("http://localhost:8888/kvs/delete/*/yay")
                .send()
                .await
                .unwrap();
            assert_eq!(resp.status(), 200);
            resp.bytes().await
        })?;
        assert_eq!(r.len(), 0); // returns nothing
                                // ensure that nothing exists with get
        let r = rt.block_on(async {
            let resp = client
                .get("http://localhost:8888/kvs/get/ghi/yay")
                .send()
                .await
                .unwrap();
            assert_eq!(resp.status(), 200);
            resp.bytes().await
        })?;
        assert_eq!(r.len(), 0);

        stop_webserver();
        Ok(())
    }

    #[test]
    #[serial_test::serial]
    fn test_api_to_abi() -> anyhow::Result<()> {
        del_attributes(&None, &None);
        init_webserver(8888)?;
        sleep(Duration::from_millis(100));
        set_status(PipelineStatus::Running)?;
        let ttl_attribute_set = AttributeSet::from(vec![Attribute::persistent(
            "jkl",
            "yay",
            vec![],
            &None,
            false,
        )]);

        let attribute_set = AttributeSet::from(vec![Attribute::persistent(
            "ghi",
            "yay",
            vec![],
            &None,
            false,
        )]);

        // set without ttl
        let rt = get_or_init_async_runtime();
        let client = reqwest::Client::new();

        let r = rt.block_on(async {
            let resp = client
                .post("http://localhost:8888/kvs/set")
                .body(attribute_set.to_pb().unwrap())
                .send()
                .await
                .unwrap();
            assert_eq!(resp.status(), 200);
            resp.bytes().await
        })?;
        assert_eq!(r.len(), 0);
        let attr = get_attribute(&"ghi".to_string(), &"yay".to_string());
        assert_eq!(attr.unwrap(), attribute_set.attributes[0]);

        // set with ttl
        let _ = rt.block_on(async {
            let resp = client
                .post("http://localhost:8888/kvs/set-with-ttl/1000")
                .body(ttl_attribute_set.to_pb().unwrap())
                .send()
                .await
                .unwrap();
            assert_eq!(resp.status(), 200);
            resp.bytes().await
        })?;
        let attr = get_attribute(&"jkl".to_string(), &"yay".to_string());
        assert_eq!(attr.unwrap(), ttl_attribute_set.attributes[0]);
        sleep(Duration::from_millis(1001));
        let attr = get_attribute(&"jkl".to_string(), &"yay".to_string());
        assert!(attr.is_none());

        stop_webserver();
        Ok(())
    }
}
