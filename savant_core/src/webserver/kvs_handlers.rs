use crate::primitives::attribute_set::AttributeSet;
use crate::protobuf::{from_pb, ToProtobuf};
use crate::webserver::kvs::asynchronous::{
    del_attribute, del_attributes, get_attribute, search_attributes, search_keys, set_attributes,
};
use actix_web::{get, post, web, HttpResponse};
use savant_protobuf::generated;

async fn set_attributes_with_ttl(payload: web::Bytes, ttl: Option<u64>) -> HttpResponse {
    let attribute_set = from_pb::<generated::AttributeSet, AttributeSet>(&payload);
    if let Ok(attribute_set) = attribute_set {
        set_attributes(&attribute_set, ttl).await;
        HttpResponse::Ok().finish()
    } else {
        HttpResponse::BadRequest().finish()
    }
}

#[post("/kvs/set-with-ttl/{ttl}")]
async fn set_handler_ttl(payload: web::Bytes, ttl: web::Path<u64>) -> HttpResponse {
    set_attributes_with_ttl(payload, Some(ttl.into_inner())).await
}

#[post("/kvs/set")]
async fn set_handler(payload: web::Bytes) -> HttpResponse {
    set_attributes_with_ttl(payload, None).await
}

#[post("/kvs/purge/{ns}/{name}")]
async fn purge_handler(path: web::Path<(String, String)>) -> HttpResponse {
    let (ns, name) = path.into_inner();
    del_attributes(&Some(ns), &Some(name)).await;
    HttpResponse::Ok().finish()
}

#[post("/kvs/delete-single/{ns}/{name}")]
async fn delete_single_handler(path: web::Path<(String, String)>) -> HttpResponse {
    let (ns, name) = path.into_inner();
    let attr_opt = del_attribute(&ns, &name).await;
    if attr_opt.is_none() {
        return HttpResponse::Ok().finish();
    }
    let pb = attr_opt.unwrap().to_pb();
    if let Ok(pb) = pb {
        HttpResponse::Ok().body(pb)
    } else {
        HttpResponse::InternalServerError().finish()
    }
}

#[get("/kvs/search/{ns}/{name}")]
async fn search_handler(path: web::Path<(String, String)>) -> HttpResponse {
    let (ns, name) = path.into_inner();
    let attr_set = search_attributes(&Some(ns), &Some(name)).await;
    let pb = attr_set.to_pb();
    if let Ok(pb) = pb {
        HttpResponse::Ok().body(pb)
    } else {
        HttpResponse::InternalServerError().finish()
    }
}

#[get("/kvs/search-keys/{ns}/{name}")]
async fn search_keys_handler(path: web::Path<(String, String)>) -> HttpResponse {
    let (ns, name) = path.into_inner();
    let keys = search_keys(&Some(ns), &Some(name)).await;
    HttpResponse::Ok().json(keys)
}

#[get("/kvs/get/{ns}/{name}")]
async fn get_handler(path: web::Path<(String, String)>) -> HttpResponse {
    let (ns, name) = path.into_inner();
    let attr_opt = get_attribute(&ns, &name).await;
    if attr_opt.is_none() {
        return HttpResponse::Ok().finish();
    }
    let pb = attr_opt.unwrap().to_pb();
    if let Ok(pb) = pb {
        HttpResponse::Ok().body(pb)
    } else {
        HttpResponse::InternalServerError().finish()
    }
}

#[cfg(test)]
mod tests {
    use crate::get_or_init_async_runtime;
    use crate::primitives::attribute_set::AttributeSet;
    use crate::primitives::Attribute;
    use crate::protobuf::{from_pb, ToProtobuf};
    use crate::webserver::kvs::synchronous::get_attribute;
    use crate::webserver::kvs::synchronous::set_attributes;
    use crate::webserver::{init_webserver, set_status, stop_webserver, PipelineStatus};
    use savant_protobuf::generated;
    use std::thread::sleep;
    use std::time::Duration;

    #[test]
    #[serial_test::serial]
    fn test_attributes_abi_to_api() -> anyhow::Result<()> {
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
        set_attributes(&ttl_attribute_set, Some(500));
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
        sleep(Duration::from_millis(501));

        let r = reqwest::blocking::get("http://localhost:8888/kvs/search-keys/*/*")?;
        assert_eq!(r.status(), 200);
        let result: Vec<(String, String)> = r.json()?;
        assert_eq!(result, vec![("ghi".to_string(), "yay".to_string())]);

        let r = reqwest::blocking::get("http://localhost:8888/kvs/search/*/*")?;
        assert_eq!(r.status(), 200);
        let binary = r.bytes()?;
        let res_attribute_set = from_pb::<generated::AttributeSet, AttributeSet>(&binary)?;
        assert_eq!(res_attribute_set.attributes, attribute_set.attributes);

        let r = reqwest::blocking::get("http://localhost:8888/kvs/get/ghi/yay")?;
        assert_eq!(r.status(), 200);
        let binary = r.bytes()?;
        let res_attribute = from_pb::<generated::Attribute, Attribute>(&binary)?;
        assert_eq!(res_attribute, attribute_set.attributes[0]);

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
        let res_attribute = from_pb::<generated::Attribute, Attribute>(&r)?;
        assert_eq!(res_attribute, attribute_set.attributes[0]);

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
        assert_eq!(r.len(), 0);

        // set again and purge
        set_attributes(&attribute_set, None);
        let r = rt.block_on(async {
            let resp = client
                .post("http://localhost:8888/kvs/purge/*/yay")
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
                .post("http://localhost:8888/kvs/set-with-ttl/500")
                .body(ttl_attribute_set.to_pb().unwrap())
                .send()
                .await
                .unwrap();
            assert_eq!(resp.status(), 200);
            resp.bytes().await
        })?;
        let attr = get_attribute(&"jkl".to_string(), &"yay".to_string());
        assert_eq!(attr.unwrap(), ttl_attribute_set.attributes[0]);
        sleep(Duration::from_millis(501));
        let attr = get_attribute(&"jkl".to_string(), &"yay".to_string());
        assert!(attr.is_none());

        stop_webserver();
        Ok(())
    }
}
