import { useEffect, useState } from 'react'
import './App.css'

const API = '/api'

function MovieCard({ movie, index, mode }) {
  const score =
    mode === 'popular'
      ? movie.weighted_score
      : mode === 'similar'
        ? movie.similarity
        : movie.personalized_score ?? movie.weighted_score

  const scoreLabel =
    mode === 'popular' ? 'Match signal' : mode === 'similar' ? 'Similarity' : 'Personalized fit'

  return (
    <article className="movie-card">
      <div className={`movie-poster poster-${index % 6}`}>
        <span className="poster-rank">{String(index + 1).padStart(2, '0')}</span>
        <span className="poster-glyph" aria-hidden="true">✦</span>
      </div>
      <div className="movie-info">
        <div className="movie-meta">
          <span>{movie.genres?.split('|')[0] || 'Featured pick'}</span>
          <span className="score-pill">{Number(score ?? 0).toFixed(2)}</span>
        </div>
        <h3>{movie.title}</h3>
        <p className="movie-genres">{movie.genres || 'Genre profile unavailable'}</p>
        <div className="movie-score">
          <span>{scoreLabel}</span>
          <strong>{mode === 'popular' ? `${Number(score ?? 0).toFixed(3)} pts` : `${Math.round(Number(score ?? 0) * 100)}%`}</strong>
        </div>
        {mode === 'popular' && (
          <p className="movie-extra">
            ★ {Number(movie.average_rating ?? 0).toFixed(1)} average · {Number(movie.rating_count ?? 0).toLocaleString()} ratings
          </p>
        )}
      </div>
    </article>
  )
}

function MovieGrid({ movies, mode }) {
  if (!movies.length) return <p className="helper-text">No titles matched this signal yet.</p>
  return (
    <div className="movie-grid">
      {movies.map((movie, index) => (
        <MovieCard key={`${mode}-${movie.movieId}`} movie={movie} index={index} mode={mode} />
      ))}
    </div>
  )
}

function App() {
  const [userId, setUserId] = useState('1')
  const [recommendations, setRecommendations] = useState([])
  const [resultUser, setResultUser] = useState(null)
  const [userLoading, setUserLoading] = useState(false)
  const [userError, setUserError] = useState('')
  const [popular, setPopular] = useState([])
  const [popularLoading, setPopularLoading] = useState(true)
  const [popularError, setPopularError] = useState('')
  const [movieId, setMovieId] = useState('318')
  const [similarMovies, setSimilarMovies] = useState([])
  const [inputMovie, setInputMovie] = useState(null)
  const [similarLoading, setSimilarLoading] = useState(false)
  const [similarError, setSimilarError] = useState('')

  async function fetchJson(path) {
    const response = await fetch(`${API}${path}`)
    const data = await response.json()
    if (!response.ok) throw new Error(data.detail || 'The request failed.')
    return data
  }

  async function loadPopular() {
    setPopularLoading(true)
    setPopularError('')
    try {
      const data = await fetchJson('/recommendations/popular?limit=10')
      setPopular(data.recommendations || [])
    } catch (err) {
      setPopularError(err.message || 'Could not load popular movies.')
    } finally {
      setPopularLoading(false)
    }
  }

  async function getInitialUserRecommendations() {
    setUserLoading(true)
    setUserError('')
    try {
      const data = await fetchJson('/recommendations/user/1?limit=10')
      setRecommendations(data.recommendations || [])
      setResultUser(data)
    } catch (err) {
      setUserError(err.message || 'Could not load initial recommendations.')
    } finally {
      setUserLoading(false)
    }
  }

  async function getRecommendations(event) {
    event.preventDefault()
    const id = Number(userId)
    if (!Number.isInteger(id) || id <= 0) {
      setUserError('Please enter a valid positive user ID.')
      return
    }
    setUserLoading(true)
    setUserError('')
    setRecommendations([])
    setResultUser(null)
    try {
      const data = await fetchJson(`/recommendations/user/${id}?limit=10`)
      setRecommendations(data.recommendations || [])
      setResultUser(data)
    } catch (err) {
      setUserError(err.message || 'Could not load personalized recommendations.')
    } finally {
      setUserLoading(false)
    }
  }

  async function getSimilarMovies(event) {
    event.preventDefault()
    const id = Number(movieId)
    if (!Number.isInteger(id) || id <= 0) {
      setSimilarError('Enter a valid positive MovieLens movie ID.')
      return
    }
    setSimilarLoading(true)
    setSimilarError('')
    setSimilarMovies([])
    setInputMovie(null)
    try {
      const data = await fetchJson(`/recommendations/movie/${id}?limit=10`)
      setSimilarMovies(data.recommendations || [])
      setInputMovie(data.input_movie || null)
    } catch (err) {
      setSimilarError(err.message || 'Could not load similar movies.')
    } finally {
      setSimilarLoading(false)
    }
  }

  // These requests intentionally run once when the workspace opens.
  /* eslint-disable react-hooks/set-state-in-effect, react-hooks/exhaustive-deps */
  useEffect(() => {
    loadPopular()
    getInitialUserRecommendations()
  }, [])
  /* eslint-enable react-hooks/set-state-in-effect, react-hooks/exhaustive-deps */

  return (
    <div className="app">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="RecoLab home">
          <span className="brand-mark">R</span>
          <span>
            <strong>RecoLab</strong>
            <small>PERSONALIZATION ENGINE</small>
          </span>
        </a>
        <div className="system-status"><i /> ML Recommendation Studio</div>
      </header>

      <main id="top" className="main-content">
        <section className="hero">
          <div className="hero-copy">
            <p className="eyebrow">RECOMMENDATION INTELLIGENCE</p>
            <h1>Discover what you'll love.</h1>
            <p className="hero-subtitle">Explore popular films, personalized picks and movies similar to your favourites, powered by MovieLens rating history and item-based collaborative filtering.</p>
          </div>
        </section>

        <section className="stats-grid" aria-label="Model statistics">
          <article className="stat-card"><span className="stat-index">01</span><p>Training interactions</p><h2>20.06M</h2><small>MovieLens 25M · training split</small></article>
          <article className="stat-card"><span className="stat-index">02</span><p>Retrieval catalog</p><h2>11,464</h2><small>Eligible titles in the catalog</small></article>
          <article className="stat-card accent-stat"><span className="stat-index">03</span><p>Current strategy</p><h2>Item similarity</h2><small>Collaborative filtering · active</small></article>
        </section>

        <section id="discover" className="content-card section-block">
          <div className="section-heading">
            <div><p className="eyebrow">THE CROWD SIGNAL</p><h2>Popular right now</h2><p>High-confidence picks from the collective rating graph.</p></div>
            <button className="ghost-button" onClick={loadPopular} disabled={popularLoading}>{popularLoading ? 'Syncing...' : 'Refresh signal ↻'}</button>
          </div>
          {popularLoading && <p className="message">Syncing the latest popularity signal...</p>}
          {popularError && <p className="error-message" role="alert">{popularError}</p>}
          {!popularLoading && !popularError && <MovieGrid movies={popular} mode="popular" />}
        </section>

        <section id="personalize" className="content-card section-block personalize-block">
          <div className="section-heading">
            <div><p className="eyebrow">YOUR SIGNAL</p><h2>Build your personal feed</h2><p>Tell Nexus who you are. We’ll surface the titles you haven’t found yet.</p></div>
            <span className="section-number">02 / 03</span>
          </div>
          <form className="recommendation-form" onSubmit={getRecommendations}>
            <label htmlFor="userId">MovieLens user ID</label>
            <div className="form-row">
              <input id="userId" type="number" min="1" step="1" value={userId} onChange={(event) => setUserId(event.target.value)} placeholder="e.g. 1" required />
              <button type="submit" disabled={userLoading}>{userLoading ? 'Reading your signal...' : 'Generate my feed ↗'}</button>
            </div>
            <p className="helper-text">Use any MovieLens user ID. Try 1, 2, 3, or 4.</p>
            <div className="demo-user-links" aria-label="Available demo profiles">
              {[1, 2, 3, 4].map((demoId) => (
                <button key={demoId} type="button" onClick={() => setUserId(String(demoId))}>
                  User {demoId}
                </button>
              ))}
            </div>
          </form>
          {userLoading && <p className="message" role="status">Mapping your taste profile to the catalog...</p>}
          {userError && <p className="error-message" role="alert">{userError}</p>}
          {resultUser && !userLoading && <section className="results-section">
            <div className="results-heading"><div><p className="eyebrow">SIGNAL MATCHED</p><h2>Selected for user {resultUser.userId}</h2><p className="helper-text">{resultUser.liked_movies_used != null ? `Based on ${resultUser.liked_movies_used} liked movies` : `Mode: ${resultUser.recommendation_type || 'fallback'}`}</p></div><span className="result-count">{recommendations.length} titles</span></div>
            <MovieGrid movies={recommendations} mode="personalized" />
          </section>}
        </section>

        <section id="similar" className="content-card section-block similar-block">
          <div className="section-heading">
            <div><p className="eyebrow">THE SIMILARITY LAB</p><h2>Follow a thread</h2><p>Drop in a movie ID and trace the titles connected by rating patterns.</p></div>
            <span className="section-number">03 / 03</span>
          </div>
          <form className="recommendation-form" onSubmit={getSimilarMovies}>
            <label htmlFor="movieId">MovieLens movie ID</label>
            <div className="form-row"><input id="movieId" type="number" min="1" step="1" value={movieId} onChange={(event) => setMovieId(event.target.value)} placeholder="e.g. 318" required /><button type="submit" disabled={similarLoading}>{similarLoading ? 'Tracing connections...' : 'Find similar titles ↗'}</button></div>
            <p className="helper-text">Start with 318 — The Shawshank Redemption.</p>
          </form>
          {similarLoading && <p className="message" role="status">Tracing connections through the rating graph...</p>}
          {similarError && <p className="error-message" role="alert">{similarError}</p>}
          {inputMovie && !similarLoading && <section className="results-section">
            <div className="results-heading"><div><p className="eyebrow">THREAD FOUND</p><h2>Because you selected {inputMovie.title}</h2><p className="helper-text">Source movie ID: {inputMovie.movieId}</p></div><span className="result-count">{similarMovies.length} titles</span></div>
            <MovieGrid movies={similarMovies} mode="similar" />
          </section>}
        </section>

        <footer><span className="footer-mark">n</span><span>NEXUS / LARGE-SCALE RECOMMENDATION ENGINE</span><span>Built on collaborative signals</span></footer>
      </main>
    </div>
  )
}

export default App
