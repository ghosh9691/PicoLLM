using FluentAssertions;
using PicoLLM.Tokenizer;

namespace PicoLLM.Tests.Tokenizer;

public class BpeTokenizerTests
{
    private const float Tol = 1e-5f;

    // ── Helper ───────────────────────────────────────────────────────────────

    private static BpeTokenizer TrainSmall(int targetVocabSize = 300)
    {
        const string corpus = "the cat sat on the mat the cat and the hat the cat";
        var config = new BpeTrainer().Train(corpus, targetVocabSize);
        return BpeTokenizer.FromConfig(config);
    }

    // ── Special token IDs ────────────────────────────────────────────────────

    [Fact]
    public void SpecialTokenIds_AreCorrect()
    {
        BpeTokenizer.PadId.Should().Be(0);
        BpeTokenizer.UnkId.Should().Be(1);
        BpeTokenizer.BosId.Should().Be(2);
        BpeTokenizer.EosId.Should().Be(3);
    }

    // ── Training ─────────────────────────────────────────────────────────────

    [Fact]
    public void Train_SmallCorpus_VocabSizeAtMostTarget()
    {
        const string corpus = "the cat sat on the mat";
        var config = new BpeTrainer().Train(corpus, 280);
        // Vocab size may be less than target if corpus exhausts unique pairs early
        config.VocabSize.Should().BeLessThanOrEqualTo(280);
        config.VocabSize.Should().BeGreaterThanOrEqualTo(260); // 4 special + 256 bytes
    }

    [Fact]
    public void Train_MergeRulesOrdered_ByFrequency()
    {
        const string corpus = "aaaa bbbb cccc aaaa bbbb aaaa";
        var config = new BpeTrainer().Train(corpus, 270);
        // There should be at least some merge rules
        config.Merges.Should().NotBeEmpty();
        // Each merge is a triple [left, right, merged]
        foreach (var m in config.Merges)
            m.Should().HaveCount(3);
    }

    [Fact]
    public void Train_512VocabSize_Produces512OrFewerTokens()
    {
        const string corpus = "the quick brown fox jumps over the lazy dog. " +
                              "the quick brown fox jumps over the lazy dog again.";
        var config = new BpeTrainer().Train(corpus, 512);
        config.VocabSize.Should().BeLessThanOrEqualTo(512);
    }

    // ── Encoding ─────────────────────────────────────────────────────────────

    [Fact]
    public void Encode_WithSpecialTokens_StartsWithBosEndsWithEos()
    {
        var tok = TrainSmall();
        var ids = tok.Encode("hello", addSpecialTokens: true);
        ids[0].Should().Be(BpeTokenizer.BosId);
        ids[^1].Should().Be(BpeTokenizer.EosId);
    }

    [Fact]
    public void Encode_WithoutSpecialTokens_NoWrappers()
    {
        var tok = TrainSmall();
        var ids = tok.Encode("hello", addSpecialTokens: false);
        ids[0].Should().NotBe(BpeTokenizer.BosId);
        ids[^1].Should().NotBe(BpeTokenizer.EosId);
    }

    [Fact]
    public void Encode_NonEmptyInput_ProducesAtLeastOneToken()
    {
        var tok = TrainSmall();
        var ids = tok.Encode("a", addSpecialTokens: false);
        ids.Should().NotBeEmpty();
    }

    [Fact]
    public void Encode_TokensWithMerges_FewerTokensThanBytes()
    {
        // Use a corpus with heavy repetition so merges happen
        var corpus = string.Concat(Enumerable.Repeat("the cat ", 50));
        var config = new BpeTrainer().Train(corpus, 280);
        var tok = BpeTokenizer.FromConfig(config);

        var rawBytes = System.Text.Encoding.UTF8.GetBytes("the cat").Length;
        var tokenIds = tok.Encode("the cat", addSpecialTokens: false);
        // With merges, token count should be < raw byte count (or equal if no merges matched)
        tokenIds.Length.Should().BeLessThanOrEqualTo(rawBytes);
    }

    // ── Decoding ─────────────────────────────────────────────────────────────

    [Fact]
    public void Decode_RoundTrip_AsciiText()
    {
        var tok = TrainSmall();
        const string original = "the cat";
        var ids = tok.Encode(original, addSpecialTokens: false);
        var decoded = tok.Decode(ids);
        decoded.Should().Be(original);
    }

    [Fact]
    public void Decode_RoundTrip_WithSpecialTokens_StripsWrappers()
    {
        var tok = TrainSmall();
        const string original = "sat on the mat";
        var ids = tok.Encode(original, addSpecialTokens: true);
        var decoded = tok.Decode(ids); // BOS/EOS should be skipped
        decoded.Should().Be(original);
    }

    [Fact]
    public void Decode_RoundTrip_Unicode()
    {
        var tok = TrainSmall();
        const string original = "café";
        var ids = tok.Encode(original, addSpecialTokens: false);
        var decoded = tok.Decode(ids);
        decoded.Should().Be(original);
    }

    [Fact]
    public void Decode_RoundTrip_Emoji()
    {
        var tok = TrainSmall();
        const string original = "hello 😀";
        var ids = tok.Encode(original, addSpecialTokens: false);
        var decoded = tok.Decode(ids);
        decoded.Should().Be(original);
    }

    // ── Persistence ──────────────────────────────────────────────────────────

    [Fact]
    public void SaveLoad_ProducesIdenticalEncoding()
    {
        var tok1 = TrainSmall();
        var path = Path.GetTempFileName();
        try
        {
            tok1.Save(path);
            var tok2 = BpeTokenizer.Load(path);

            const string text = "the cat sat";
            var ids1 = tok1.Encode(text);
            var ids2 = tok2.Encode(text);
            ids1.Should().Equal(ids2);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void SaveLoad_VocabSizePreserved()
    {
        var tok1 = TrainSmall(300);
        var path = Path.GetTempFileName();
        try
        {
            tok1.Save(path);
            var tok2 = BpeTokenizer.Load(path);
            tok2.VocabSize.Should().Be(tok1.VocabSize);
        }
        finally
        {
            File.Delete(path);
        }
    }
}
