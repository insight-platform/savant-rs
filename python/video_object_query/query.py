from savant_rs.video_object_query import FloatExpression as FE, \
    IntExpression as IE, \
    StringExpression as SE, \
    Query as Q

and_ = Q.and_
or_ = Q.or_
not_ = Q.not_


gt = IE.gt
lt = IE.lt
eq = IE.eq
fgt = FE.gt

q = and_(
    Q.creator(SE.one_of('savant', 'deepstream')),
    Q.label(SE.one_of('person', 'cyclist')),
    and_(
        or_(
            not_(Q.parent_defined()),
            or_(
                Q.parent_id(IE.one_of(0, 1, 2)),
                Q.parent_id(gt(10))
            )
        )
    ),
    Q.attributes_jmes_query("[?(name=='test' && creator=='test')]"),
    Q.confidence(FE.gt(0.5)),
    Q.box_height(FE.gt(100)),
)

print("------------------------")
print("Condensed JSON:")
print("------------------------")
print(q.json)

print("------------------------")
print("Pretty JSON:")
print("------------------------")
print(q.json_pretty)

print("------------------------")
print("YAML:")
print("------------------------")
print(q.yaml)

q2 = Q.from_json(q.json)
assert q.json == q2.json

q3 = Q.from_yaml(q.yaml)
assert q3.yaml == q.yaml

