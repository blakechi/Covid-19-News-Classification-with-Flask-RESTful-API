import functools

# Reference from: https://github.com/yinwenpeng/BenchmarkingZeroShot/blob/master/src/train_yahoo.py
# Template
hypothesis_template = "This text {}."

# Top class
top_class_to_hypothesis = {
    "Containment and Closure Policies": [
        'is related with containment and closure policy from governments in the pandemic',
    ],
    "Economic Policies": [
        'is related with economic policy from governments in the pandemic', 
    ],
    "Health System Policies": [
        'is related with health system policy from governments in the pandemic',
    ],
    "Miscellaneous Policies": [
        'is related with miscellaneous policy from governments in the pandemic', 
    ]
}

# Sub class
sub_class_to_hypothesis = {
    "Containment and Closure Policies": [
        'record closing of schools and universities',                                   
        'record closing of workplaces',                                   
        'record cancellation of public events',                                   
        'record limit on gatherings',                                   
        'record closing of public transport',                                   
        'record order to "shelter-in-place" and otherwise confine to the home',                                   
        'record restriction on internal movement between cities or regions',                                   
        'record restriction on international travel for foreign travellers, not citizens',                                   
        # 'school closing', 
        # 'workspace closing', 
        # 'public event cancellation', 
        # 'restrictions on gatherings', 
        # 'public transport closing', 
        # 'stay at home requirement', 
        # 'restrictions on internel movement', 
        # 'international travel control'
    ],
    "Economic Policies": [
        'record if the government is providing direct cash payments to people who lose their jobs or cannot work',
        'record if the government is freezing financial obligations for households, like stopping loan repayments, preventing services like water from stopping, or banning evictions',
        'announced economic stimulus spending',
        'announced offer of Covid-19 related aid spending to other countries',
        # 'income support', 
        # 'debt or contract relief', 
        # 'fiscal measurements', 
        # 'international support in the pandemic',
    ],
    "Health System Policies": [
        'record presence of public info campaigns',
        'record government policy on who has access to PCR testing instead of antibody test',
        'record government policy on contact tracing after a positive diagnosis',
        'announced short term spending on healthcare system, eg hospitals, masks, etc',
        'announced public spending on Covid-19 vaccine development',
        'record policy on the use of facial coverings outside the home',
        'record policy for vaccine delivery for different groups',
        # 'public health compaigns', 
        # 'testing policy',
        # 'contact tracing',
        # 'emergency investment in health care',
        # 'investment in vaccines' 
        # 'facial coverings',
        # 'vaccination policy in the pandemic',
    ],
}

top_hypothesis = functools.reduce(lambda a, b: a + b, top_class_to_hypothesis.values())
sub_hypothesis = functools.reduce(lambda a, b: a + b, sub_class_to_hypothesis.values())

#
hypothesis_candidate = top_hypothesis + sub_hypothesis

#
labels_to_indice = { label: idx for idx, label in enumerate(top_hypothesis) }
for top_idx, top_key in enumerate(top_class_to_hypothesis.keys()):  # for all except Miscellaneous: add sub-hypothesis
    if top_key != "Miscellaneous Policies":
        sub_label = sub_class_to_hypothesis[top_key]
        labels_to_indice.update(
            {
                label: [top_idx, sub_idx] for sub_idx, label in enumerate(sub_label)
            }
        )

#
top_class = list(top_class_to_hypothesis.keys())

#
sub_class = [
    [
        "School Closing",
        "Workplace Closing",
        "Public Event Cancellation",
        "Restrictions on Gatherings",
        "Public Transport Closing",
        "Stay at Home Requirements",
        'Restrictions on Internel Movement', 
        'International Travel Control'
    ],
    [
        "Income Support",
        "Debt or Contract Relief",
        "Fiscal Measurements",
        "International Suport in the Pandemic",
    ],
    [
        "Public Health Campaigns",
        "Testing Policy",
        "Contact Tracing",
        "Emergency Investment in Health Care",
        "Investment in Vaccines",
        "Facial Coverings",
        "Vaccination Policy in the Pandemic",
    ]
]

#
output_template = [
    {
        'class': 'root',
        'children': [
            {
                'class': top_class_name,
                'percentage': 0,
                'children': [
                    {
                        'class': sub_class_name,
                        'percentage': 0
                    }
                    for sub_class_name in sub_class[top_idx]
                ]
            }
            for top_idx, top_class_name in enumerate(top_class) if top_idx < 3
        ],
    }
]
output_template[0]['children'].append(
    {
        'class': "Miscellaneous Policies",
        'percentage': 0,
    }
)

# A hyperparametor for the softmax at the output
temperature = 0.1


if __name__ == "__main__":
    import json
    
    
    print(json.dumps(hypothesis_candidate, indent=4))
    print(json.dumps(labels_to_indice, indent=4))
    print(json.dumps(output_template, indent=4))