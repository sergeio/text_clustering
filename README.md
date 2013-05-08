text-clustering
===============

An implementation of textual clustering, using k-means for clustering, and
cosine similarity ([link](https://github.com/sergeio/text_comparer)) as the
distance metric.

Ideal use case:
```python
In [1]: from vectorizer import cluster_paragraphs

# define text variables

In [2]: cluster_paragraphs([
   ...:     text_about_thing_a,
   ...:     text_about_thing_b,
   ...:     text_about_thing_a2,
   ...:     text_about_thing_a3,
   ...:     text_about_thing_b2,
   ...: ], num_clusters=2)
Out[2]: [
   ...:     [text_about_thing_a, text_about_thing_a2, text_about_thing_a3],
   ...:     [text_about_thing_b, text_about_thing_b2],
   ...: ]
```

You give the function a list with text, and it groups them into clusters by
analyzing the content of each string.

More documentation to come! <- this could be a lie.
