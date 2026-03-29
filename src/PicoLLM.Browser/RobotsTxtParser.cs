using System.Collections.Concurrent;

namespace PicoLLM.Browser;

/// <summary>
/// Fetches and caches robots.txt per domain for the lifetime of a session.
/// Parses <c>User-agent</c> and <c>Disallow</c> directives and checks whether
/// a given URL path is allowed for the PicoBrowser user agent.
/// </summary>
public sealed class RobotsTxtParser
{
    private readonly HttpClient _client;

    /// <summary>Per-domain cache: domain → list of disallowed path prefixes.</summary>
    private readonly ConcurrentDictionary<string, IReadOnlyList<string>> _cache = new();

    internal RobotsTxtParser(HttpClient client)
    {
        _client = client;
    }

    /// <summary>
    /// Returns true if fetching <paramref name="uri"/> is permitted by robots.txt,
    /// false if it is explicitly disallowed. When robots.txt cannot be fetched the
    /// request is allowed (fail-open).
    /// </summary>
    public async Task<bool> IsAllowedAsync(Uri uri, CancellationToken cancellationToken = default)
    {
        string domain = uri.GetLeftPart(UriPartial.Authority); // scheme://host:port

        if (!_cache.TryGetValue(domain, out var disallowedPaths))
        {
            disallowedPaths = await FetchDisallowedPathsAsync(domain, cancellationToken)
                .ConfigureAwait(false);
            _cache[domain] = disallowedPaths;
        }

        string path = uri.PathAndQuery;
        foreach (string prefix in disallowedPaths)
        {
            if (path.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
                return false;
        }

        return true;
    }

    private async Task<IReadOnlyList<string>> FetchDisallowedPathsAsync(
        string domain, CancellationToken cancellationToken)
    {
        try
        {
            string robotsUrl = domain + "/robots.txt";
            using var response = await _client.GetAsync(robotsUrl, cancellationToken).ConfigureAwait(false);
            if (!response.IsSuccessStatusCode)
                return [];

            string text = await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
            return Parse(text);
        }
        catch
        {
            // Fail-open: if robots.txt can't be fetched, allow everything
            return [];
        }
    }

    /// <summary>
    /// Parses a robots.txt body and returns the Disallow prefixes that apply to
    /// any user-agent (<c>*</c>) or specifically to <c>PicoBrowser</c>.
    /// </summary>
    internal static IReadOnlyList<string> Parse(string content)
    {
        var disallowed = new List<string>();
        bool applicable = false; // are we inside a matching User-agent block?

        foreach (string rawLine in content.Split('\n'))
        {
            string line = rawLine.Trim();

            // Strip inline comments
            int commentIdx = line.IndexOf('#');
            if (commentIdx >= 0)
                line = line[..commentIdx].Trim();

            if (line.Length == 0)
            {
                // Blank line resets the User-agent scope
                applicable = false;
                continue;
            }

            if (line.StartsWith("User-agent:", StringComparison.OrdinalIgnoreCase))
            {
                string agent = line["User-agent:".Length..].Trim();
                applicable = agent == "*" ||
                             agent.Equals("PicoBrowser", StringComparison.OrdinalIgnoreCase);
                continue;
            }

            if (applicable && line.StartsWith("Disallow:", StringComparison.OrdinalIgnoreCase))
            {
                string path = line["Disallow:".Length..].Trim();
                if (!string.IsNullOrEmpty(path))
                    disallowed.Add(path);
            }
        }

        return disallowed;
    }
}
