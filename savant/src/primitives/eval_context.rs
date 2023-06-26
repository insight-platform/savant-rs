use crate::utils::eval_resolvers::get_symbol_resolver;
use evalexpr::{EvalexprError, EvalexprResult, Value};

pub trait EvalWithResolvers {
    fn get_resolvers(&self) -> &'_ [String];

    fn resolve(&self, identifier: &str, argument: &Value) -> EvalexprResult<Value> {
        let res = get_symbol_resolver(identifier);
        match res {
            Some((r, executor)) => {
                if self.get_resolvers().contains(&r) {
                    executor
                        .resolve(identifier, argument)
                        .map_err(|e| EvalexprError::CustomMessage(e.to_string()))
                } else {
                    Err(EvalexprError::FunctionIdentifierNotFound(
                        identifier.to_string(),
                    ))
                }
            }
            None => Err(EvalexprError::FunctionIdentifierNotFound(
                identifier.to_string(),
            )),
        }
    }
}
