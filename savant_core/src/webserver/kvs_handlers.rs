use crate::primitives::attribute_set::AttributeSet;
use crate::protobuf::{from_pb, ToProtobuf};
use crate::webserver::kvs::asynchronous::{
    del_attribute, del_attributes, get_attribute, search_attributes, search_keys, set_attributes,
};
use actix_web::{get, post, web, HttpResponse};
use lazy_static::lazy_static;
use savant_protobuf::generated;

lazy_static! {
    static ref EMPTY_SERIALIZED_ATTRIBUTE_SET: Vec<u8> = AttributeSet::new().to_pb().unwrap();
}

async fn set_attributes_with_ttl(payload: web::Bytes, ttl: Option<u64>) -> HttpResponse {
    let attribute_set = from_pb::<generated::AttributeSet, AttributeSet>(&payload);
    if let Ok(attribute_set) = attribute_set {
        set_attributes(&attribute_set.attributes, ttl).await;
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

#[post("/kvs/delete/{ns}/{name}")]
async fn delete_handler(path: web::Path<(String, String)>) -> HttpResponse {
    let (ns, name) = path.into_inner();
    del_attributes(&Some(ns), &Some(name)).await;
    HttpResponse::Ok().finish()
}

#[post("/kvs/delete-single/{ns}/{name}")]
async fn delete_single_handler(path: web::Path<(String, String)>) -> HttpResponse {
    let (ns, name) = path.into_inner();
    let attr_opt = del_attribute(&ns, &name).await;
    if attr_opt.is_none() {
        return HttpResponse::Ok().body(EMPTY_SERIALIZED_ATTRIBUTE_SET.clone());
    }
    let attr_set = AttributeSet::from(vec![attr_opt.unwrap()]);
    let pb = attr_set.to_pb();
    if let Ok(pb) = pb {
        HttpResponse::Ok().body(pb)
    } else {
        HttpResponse::InternalServerError().finish()
    }
}

#[get("/kvs/search/{ns}/{name}")]
async fn search_handler(path: web::Path<(String, String)>) -> HttpResponse {
    let (ns, name) = path.into_inner();
    let attrs = search_attributes(&Some(ns), &Some(name)).await;
    let attr_set = AttributeSet::from(attrs);
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
        return HttpResponse::Ok().body(EMPTY_SERIALIZED_ATTRIBUTE_SET.clone());
    }
    let attr_set = AttributeSet::from(vec![attr_opt.unwrap()]);
    let pb = attr_set.to_pb();
    if let Ok(pb) = pb {
        HttpResponse::Ok().body(pb)
    } else {
        HttpResponse::InternalServerError().finish()
    }
}
