from vectorizer import cluster_paragraphs
from random import shuffle


def main():
    """Try clustering texts repeatedly to get the success rate."""
    samples = 1000
    results = {True: 0, False: 0}
    for _ in xrange(samples):
        result = try_clustering()
        results[result] += 1
    success_rate = int(100.0 * results[True] / samples)
    return 'Clustering was perfectly correct {0}% of the time.'.format(success_rate)


def cluster_randomly(texts):
    """Cluster randomly into two groups, assuming one group has 2 members."""
    return [texts[:2], texts[2:]]


def verify_results(correct1, correct2, observed1, observed2):
    """Are the observed results correct?"""
    return correct1 == observed1 or correct1 == observed2


def try_clustering():
    """Try clustering some texts.  Return whether we got the right answer."""
    text1 = """Type theory is closely related to (and in some cases overlaps with) type systems, which are a programming language feature used to reduce bugs. The types of type theory were created to avoid paradoxes in a variety of formal logics and rewrite systems and sometimes "type theory" is used to refer to this broader application."""
    text2 = """The types of type theory were invented by Bertrand Russell in response to his discovery that Gottlob Frege's version of naive set theory was afflicted with Russell's paradox. This theory of types features prominently in Whitehead and Russell's Principia Mathematica. It avoids Russell's paradox by first creating a hierarchy of types, then assigning each mathematical (and possibly other) entity to a type. Objects of a given type are built exclusively from objects of preceding types (those lower in the hierarchy), thus preventing loops."""
    text3 = """The common usage of "type theory" is when those types are used with a term rewrite system. The most famous early example is Alonzo Church's lambda calculus. Church's Theory of Types[1] helped the formal system avoid the Kleene-Rosser paradox that afflicted the original untyped lambda calculus. Church demonstrated that it could serve as a foundation of mathematics and it was referred to as a higher-order logic."""
    text4 = """A pulsar (short for pulsating radio star) is a highly magnetized, rotating neutron star that emits a beam of electromagnetic radiation. This radiation can only be observed when the beam of emission is pointing toward the Earth, much the way a lighthouse can only be seen when the light is pointed in the direction of an observer, and is responsible for the pulsed appearance of emission. Neutron stars are very dense, and have short, regular rotational periods. This produces a very precise interval between pulses that range from roughly milliseconds to seconds for an individual pulsar."""
    text5 = """The precise periods of pulsars make them useful tools. Observations of a pulsar in a binary neutron star system were used to indirectly confirm the existence of gravitational radiation."""
    texts = [text1, text2, text3, text4, text5]
    shuffle(texts)

    clusters = cluster_randomly(texts)
    # clusters = cluster_paragraphs(texts, num_clusters=2)

    correct_cluster1 = set([text1, text2, text3])
    correct_cluster2 = set([text4, text5]),
    observed_clusters = [set(clusters[0]), set(clusters[1])]
    success = verify_results(correct_cluster1,
            correct_cluster2,
            observed_clusters[0],
            observed_clusters[1])

    return success


if __name__ == '__main__':
    print main()
